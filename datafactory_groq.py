#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
مصنع بيانات ذكي يعتمد على Groq API لتحويل الأسئلة العربية إلى صيغة Micro-Engine.
يدعم تحميل البيانات من ArabicMMLU مع إمكانية خلط العينات عشوائياً.
"""

import os
import json
import time
import logging
import random
import re
import sys
from tqdm import tqdm
from groq import Groq
from datasets import load_dataset  # لتحميل البيانات من HuggingFace

# ==========================================
# 1. الإعدادات العامة
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# متغيرات البيئة
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY غير موجود. عيّنه في متغيرات البيئة.")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TEMP = float(os.environ.get("GROQ_TEMP", 0.5))
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", 7000))

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BASE_DELAY = float(os.environ.get("BASE_DELAY", 2.0))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", 3.0))
JITTER = float(os.environ.get("JITTER", 1.0))

# إعدادات البيانات
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 50))          # عدد العينات المطلوبة
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "micro_engine_train_data.jsonl")

# البذرة العشوائية لضمان تنوع العينات في كل تشغيل
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", random.randint(1, 1000000)))

# تهيئة عميل Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ تم تهيئة عميل Groq.")
except Exception as e:
    logger.error(f"❌ فشل تهيئة Groq: {e}")
    groq_client = None
    sys.exit(1)

# ==========================================
# 2. تحميل البيانات من ArabicMMLU (عشوائية)
# ==========================================
def load_arabic_mmlu(max_samples=50, seed=None):
    """تحميل عينات عشوائية من ArabicMMLU (تختلف كل مرة)."""
    if seed is None:
        seed = random.randint(1, 1000000)
    
    subjects = [
        'Physics (High School)',
        'Biology (High School)',
        'Arabic Language (High School)',
        'Arabic Language (Grammar)'
    ]
    
    data = []
    per_subject = max(1, max_samples // len(subjects))
    
    for sub in subjects:
        try:
            # حمّل مجموعة الاختبار كاملة
            ds = load_dataset("MBZUAI/ArabicMMLU", sub, split="test")
            
            # خلط الأسئلة باستخدام البذرة العشوائية (Seed)
            shuffled_ds = ds.shuffle(seed=seed)
            
            count = 0
            for item in shuffled_ds:
                if count >= per_subject:
                    break
                question = item.get("Question")
                options = [
                    item.get("Option 1"),
                    item.get("Option 2"),
                    item.get("Option 3"),
                    item.get("Option 4")
                ]
                options = [opt for opt in options if opt and str(opt).strip()]
                answer = item.get("Answer Key")
                
                if question and len(options) >= 2 and answer:
                    data.append({
                        "question": question,
                        "choices": options,
                        "answer": answer,
                        "subject": sub
                    })
                    count += 1
            logger.info(f"✅ تم تحميل {count} عينة عشوائية من {sub} (الـ Seed: {seed}).")
        except Exception as e:
            logger.warning(f"⚠️ فشل تحميل {sub}: {e}")
    
    return data

# ==========================================
# 3. قالب Micro-Engine
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
    return (
        f"<|startoftext|><|pad|><|unk|><|system|>{SYSTEM_PROMPT}<|end_context|>\n"
        f"<|startoftext|><|user|>{user_query}<|endoftext|>\n"
        f"<|assistant|><|think|>{think_content}<|end_think|>\n"
        f"<|solve|>{solve_content}<|endoftext|>"
    )

# ==========================================
# 4. استدعاء Groq مع إعادة المحاولة
# ==========================================
def call_groq_with_retry(prompt_instruction):
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
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            wait_time = BASE_DELAY * (2 ** attempt)
            if "429" in error_msg or "Rate limit" in error_msg:
                match = re.search(r'try again in ([\d.]+)s', error_msg)
                if not match:
                    match = re.search(r'(\d+\.?\d*) seconds', error_msg)
                if match:
                    wait_time = float(match.group(1)) + 0.5
            logger.warning(f"⚠️ محاولة {attempt+1} فشلت: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"⏳ الانتظار {wait_time:.2f} ثانية...")
                time.sleep(wait_time)
            else:
                raise
    return None

def extract_json(text):
    """استخراج JSON من النص باستخدام التوازن بين الأقواس."""
    start = text.find('{')
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None

# ==========================================
# 5. المعالجة الرئيسية
# ==========================================
def process_samples(samples, output_file):
    """معالجة قائمة العينات وإنتاج JSONL، مع إحصائيات عن النجاح والفشل."""
    if not samples:
        logger.warning("⚠️ لا توجد عينات للمعالجة.")
        return 0, 0

    # إنشاء مجلد المخرجات إذا لم يكن موجودًا
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    successful = 0
    failed = 0

    # فتح ملف المخرجات للإلحاق (للاستئناف)
    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(samples, desc="معالجة العينات")):
            # بناء النص الذي سيُرسل إلى النموذج
            if "question" in item:
                user_query = item["question"]
                # إضافة الخيارات إن وجدت
                if "choices" in item:
                    choices_text = "\n".join([f"{chr(65+i)}. {ch}" for i, ch in enumerate(item["choices"])])
                    user_query += f"\nالخيارات:\n{choices_text}"
                    if "answer" in item:
                        user_query += f"\nالإجابة الصحيحة: {item['answer']}"
            elif "text" in item:
                user_query = item["text"]
            else:
                logger.warning(f"⚠️ العينة {idx} لا تحتوي على مفتاح معروف، تم تخطيها.")
                failed += 1
                continue

            prompt_instruction = (
                f"قم بحل المسألة أو الإجابة على السؤال التالي ملتزماً بالأدوار الثلاثة:\n"
                f"1) التفكير الجدلي والعميق.\n"
                f"2) التخطيط للحل خطوة بخطوة.\n"
                f"3) تقديم الحل النهائي المباشر.\n\n"
                f"السؤال:\n{user_query}\n\n"
                f"⚠️ استجب بتنسيق JSON حصرياً كالتالي:\n"
                f"{{\n  \"thinking\": \"...\",\n  \"solution\": \"...\"\n}}"
            )

            try:
                raw = call_groq_with_retry(prompt_instruction)
                # استخراج JSON
                json_str = extract_json(raw)
                if not json_str:
                    raise ValueError("لا يوجد JSON صحيح.")
                # تنظيف الفواصل الزائدة
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                res = json.loads(json_str)

                think = res.get("thinking", "").strip()
                solve = res.get("solution", "").strip()
                if not think or not solve:
                    raise ValueError("محتوى فارغ.")

                final_text = format_to_micro_engine(user_query, think, solve)
                record = {
                    "text": final_text,
                    "metadata": {"index": idx, "source": "groq_enriched"}
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                successful += 1
                logger.info(f"✅ تمت معالجة العينة {idx+1}/{len(samples)}")

                # تأخير
                time.sleep(REQUEST_DELAY + random.uniform(0, JITTER))

            except Exception as e:
                logger.error(f"❌ فشل العينة {idx}: {e}")
                failed += 1

    return successful, failed

# ==========================================
# 6. عرض الخلاصة النهائية
# ==========================================
def print_summary(successful, failed, output_file):
    """طباعة خلاصة بسيطة عن عملية المعالجة."""
    file_size = 0
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
    
    print("\n" + "="*50)
    print("           🎯  ملخص تشغيل مصنع البيانات")
    print("="*50)
    print(f"✅ العينات الناجحة:   {successful}")
    print(f"❌ العينات الفاشلة:   {failed}")
    print(f"📂 إجمالي العينات:    {successful + failed}")
    print(f"📄 حجم الملف الناتج:  {file_size:,} بايت")
    print(f"📁 مسار الملف:        {output_file}")
    print("="*50)
    print("✨ انتهت المعالجة بنجاح." if failed == 0 else "⚠️ انتهت المعالجة مع بعض الأخطاء.")
    print("="*50 + "\n")

# ==========================================
# 7. الدالة الرئيسية
# ==========================================
def main():
    # 1. تحميل البيانات من ArabicMMLU مع خلط عشوائي
    logger.info(f"📂 تحميل عينات من ArabicMMLU (الـ Seed: {RANDOM_SEED})")
    samples = load_arabic_mmlu(max_samples=MAX_SAMPLES, seed=RANDOM_SEED)

    if not samples:
        logger.error("❌ لا توجد بيانات للمعالجة. تأكد من المصادر.")
        sys.exit(1)

    logger.info(f"🚀 بدء المعالجة لـ {len(samples)} عينة...")
    successful, failed = process_samples(samples, OUTPUT_FILE)
    
    # عرض الخلاصة
    print_summary(successful, failed, OUTPUT_FILE)

if __name__ == "__main__":
    main()
