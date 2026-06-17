#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
سكريبت معالجة البيانات وتحويلها إلى صيغة Micro-Engine باستخدام Groq API
مع تحسينات للتعامل مع حدود المعدل (Rate Limits) وإعادة المحاولة الذكية.
"""

import os
import json
import time
import logging
import random
import re
from tqdm import tqdm
from groq import Groq

# ==========================================
# 1. الإعدادات العامة وتسجيل المخرجات (Logging)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------
# قراءة المفاتيح من متغيرات البيئة أو استخدم قيماً افتراضية (للاختبار فقط)
# ----------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
    logger.warning("⚠️ لم يتم تعيين GROQ_API_KEY، استخدم متغير البيئة أو عدّل المفتاح في الكود.")

# نموذج Groq – يُفضل استخدام نموذج يدعم JSON mode (مثل llama3-70b-8192)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
GROQ_TEMP = float(os.environ.get("GROQ_TEMP", 0.5))
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", 2048))

# إعدادات إعادة المحاولة والتأخير
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BASE_DELAY = float(os.environ.get("BASE_DELAY", 2.0))      # التأخير الأساسي بين المحاولات
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", 3.0)) # التأخير بين الطلبات الناجحة (لتجنب الـ RPM)
JITTER = float(os.environ.get("JITTER", 1.0))              # قيمة عشوائية تضاف للتأخير

# أسماء الملفات (قم بتعديلها لتطابق أسماء ملفاتك الحقيقية)
INPUT_JSON_FILE = os.environ.get("INPUT_JSON_FILE", "your_input_data.json")
OUTPUT_DATASET_FILE = os.environ.get("OUTPUT_DATASET_FILE", "micro_engine_train_data.jsonl")

# تهيئة عميل Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ تم تهيئة عميل Groq بنجاح.")
except Exception as e:
    logger.error(f"❌ فشل تهيئة عميل Groq: {e}")
    groq_client = None

# ==========================================
# 2. القالب الهيكلي الموحد لـ Micro-Engine
# ==========================================
SYSTEM_PROMPT = (
    "أنت نظام وكلاء متعدد المهام (Multi-Agent System) يعمل على نموذج Micro-Engine. "
    "أنت تجمع بين ثلاثة أدوار: 1) المفكر الجدلي العميق (DeepSeek-R1 Style) الذي يشكك ويدقق ويجادل داخل <|think|>. "
    "2) وكيل التخطيط (Planning Agent) الذي يقسم المهام الكبيرة إلى خطوات Step 1, Step 2... "
    "3) وكيل الاستشهاد الحرفي (Citation Finder) الذي يسترجع النصوص الأصلية دون تعديل. "
    "خرجك النهائي دائماً داخل <|solve|> بغض النظر عن نوع المهمة. "
    "عند نقص البيانات، توقف واطلبها بدلاً من التخمين."
)

def format_to_micro_engine(user_query, think_content, solve_content):
    """
    تأطير البيانات بالرموز الدلالية الخاصة بالـ Micro-Engine لتجهيزها للتدريب.
    """
    formatted_text = (
        f"<|startoftext|><|pad|><|unk|><|system|>{SYSTEM_PROMPT}<|end_context|>\n"
        f"<|startoftext|><|user|>{user_query}<|endoftext|>\n"
        f"<|assistant|><|think|>{think_content}<|end_think|>\n"
        f"<|solve|>{solve_content}<|endoftext|>"
    )
    return formatted_text

# ==========================================
# 3. دوال مساعدة للتعامل مع الـ API وإعادة المحاولة
# ==========================================
def call_groq_with_retry(prompt_instruction):
    """
    استدعاء Groq API مع إعادة محاولة ذكية تستخرج وقت الانتظار من رسائل الخطأ (خاصة 429).
    """
    if not groq_client:
        raise RuntimeError("عميل Groq غير مهيأ.")

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise data creation engine that outputs strictly JSON."},
                    {"role": "user", "content": prompt_instruction}
                ],
                temperature=GROQ_TEMP,
                max_tokens=GROQ_MAX_TOKENS,
                # ملاحظة: response_format مدعوم في بعض النماذج، لكن قد يسبب مشاكل،
                # لذا نضعه في try-except أو نتركه معتمداً على تعليمات النموذج.
                # نفضل إزالة response_format والاعتماد على التعليمات النصية.
                # response_format={"type": "json_object"}  # يمكن تفعيله إذا كان النموذج يدعمه
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            wait_time = BASE_DELAY * (2 ** attempt)  # تأخير متزايد (Backoff)

            # محاولة استخراج وقت الانتظار من رسائل 429
            if "429" in error_msg or "Rate limit" in error_msg:
                # البحث عن "try again in Xs" أو "X seconds"
                match = re.search(r'try again in ([\d.]+)s', error_msg)
                if not match:
                    match = re.search(r'(\d+\.?\d*) seconds', error_msg)
                if match:
                    wait_time = float(match.group(1)) + 0.5  # نضيف نصف ثانية احتياطية

            logger.warning(f"⚠️ محاولة {attempt+1} لـ Groq فشلت: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"⏳ الانتظار {wait_time:.2f} ثانية قبل إعادة المحاولة...")
                time.sleep(wait_time)
            else:
                # آخر محاولة فشلت، نرفع الاستثناء
                raise

    # لن نصل هنا
    return None

# ==========================================
# 4. المعالجة الرئيسية
# ==========================================
def process_and_enrich_dataset(input_file, output_file):
    if not groq_client:
        logger.error("❌ لا يمكن بدء المعالجة بدون عميل API صالح.")
        return

    if not os.path.exists(input_file):
        logger.error(f"❌ لم يتم العثور على ملف المدخلات: {input_file}. تأكد من تسميته بشكل صحيح.")
        return

    logger.info(f"🟢 قراءة البيانات من {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data_items = json.load(f)
    except Exception as e:
        logger.error(f"❌ خطأ أثناء قراءة ملف JSON: {e}")
        return

    # التأكد من أن البيانات عبارة عن قائمة
    if not isinstance(data_items, list):
        logger.warning("⚠️ تحذير: ملف JSON ليس قائمة (List). سيتم محاولة معالجته كعنصر مفرد.")
        data_items = [data_items]

    logger.info(f"🚀 تم العثور على {len(data_items)} عينة. بدء المعالجة والتوليد عبر Groq...")

    # فتح ملف المخرجات في وضع الإضافة (Append) لتحقيق خصائص الاستئناف (Checkpointing)
    with open(output_file, "a", encoding="utf-8") as out_f:
        # نستخدم tqdm لعرض التقدم
        for index, item in enumerate(tqdm(data_items, desc="Processing Items")):
            # البحث عن النص في عدة مفاتيح محتملة
            user_query = (
                item.get("question") or
                item.get("text") or
                item.get("prompt") or
                item.get("input") or
                item.get("content")
            )
            if not user_query:
                logger.warning(
                    f"⚠️ العينة رقم {index} لا تحتوي على مفتاح نصي معروف. تم تخطيها. "
                    f"المفاتيح المتاحة: {list(item.keys())}"
                )
                continue

            # صياغة التعليمات لإجبار النموذج على إخراج JSON
            prompt_instruction = (
                f"قم بحل المسألة أو الإجابة على السؤال التالي ملتزماً بالأدوار الثلاثة لنظام الوكلاء:\n"
                f"1) التفكير الجدلي والعميق والنقد الذاتي.\n"
                f"2) التخطيط للحل خطوة بخطوة.\n"
                f"3) تقديم الحل النهائي المباشر المستقر.\n\n"
                f"المسألة/السؤال:\n{user_query}\n\n"
                f"⚠️ يجب أن تكون استجابتك بتنسيق JSON حصراً ودون أي نصوص إضافية خارج الأقواس، كالتالي:\n"
                f"{{\n"
                f"  \"thinking\": \"اكتب هنا كل مجريات التفكير الجدلي والخطوات والتخطيط العميق...\",\n"
                f"  \"solution\": \"اكتب هنا الحل النهائي الحتمي المباشر...\"\n"
                f"}}"
            )

            success = False
            try:
                raw_content = call_groq_with_retry(prompt_instruction)

                # محاولة استخراج JSON من النص (قد يحتوي على Markdown أو نصوص إضافية)
                # نستخدم regex لاستخراج أول كائن JSON صحيح
                json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
                if not json_match:
                    raise ValueError("لم يتم العثور على JSON صحيح في الاستجابة.")

                json_str = json_match.group(1)
                # تنظيف بسيط: إزالة الفواصل الزائدة قبل الأقواس
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

                res_json = json.loads(json_str)

                think_content = res_json.get("thinking", "").strip()
                solve_content = res_json.get("solution", "").strip()

                if not think_content or not solve_content:
                    raise ValueError("محتوى التفكير أو الحل فارغ في استجابة الـ API.")

                # صياغة النص بالتنسيق النهائي
                final_sample_text = format_to_micro_engine(user_query, think_content, solve_content)

                # حفظ العينة بصيغة JSONL
                output_entry = {
                    "text": final_sample_text,
                    "metadata": {
                        "original_index": index,
                        "source_file": input_file
                    }
                }
                out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                out_f.flush()  # كتابة فورية

                success = True

            except Exception as e:
                logger.error(f"❌ فشل معالجة العينة {index}: {e}")

            # تأخير بين الطلبات الناجحة (مع جيتر) لتجنب تجاوز حدود المعدل
            if success:
                delay = REQUEST_DELAY + random.uniform(0, JITTER)
                time.sleep(delay)

    logger.info(f"✨ اكتملت المعالجة! تم حفظ البيانات النهائية المجهزة للتدريب في: {output_file}")

# ==========================================
# 5. نقطة الانطلاق لتشغيل السكريبت
# ==========================================
if __name__ == "__main__":
    # يمكن تمرير أسماء الملفات كوسائط أو استخدام القيم الافتراضية
    process_and_enrich_dataset(INPUT_JSON_FILE, OUTPUT_DATASET_FILE)
