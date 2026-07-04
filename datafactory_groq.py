#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
مصنع بيانات ذكي يعتمد على Groq API لتحويل الأسئلة العربية إلى صيغة Micro-Engine.
يدعم تحميل البيانات من ArabicMMLU مع إمكانية خلط العينات عشوائياً.
تم تصحيح أخطاء JSON وحذف الإجابات الصحيحة من الطلب.
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
from datasets import load_dataset

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

GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")  # نموذج أكثر استقراراً في JSON
GROQ_TEMP = float(os.environ.get("GROQ_TEMP", 0.4))
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", 7700))

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BASE_DELAY = float(os.environ.get("BASE_DELAY", 2.0))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", 3.0))
JITTER = float(os.environ.get("JITTER", 1.0))

MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 200))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "micro_engine_train_data.jsonl")

# البذرة العشوائية لتنويع العينات
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
def load_arabic_mmlu(max_samples=200, seed=None):
    """تحميل عينات عشوائية من ArabicMMLU (تختلف كل مرة)."""
    if seed is None:
        seed = random.randint(1, 1000000)
    
    subjects = [
        'Physics (High School)',
        'Biology (High School)',
        'Arabic Language (High School)',
        'Arabic Language (Grammar)'
        
'Geography (High School)'

'Islamic Studies (Primary School)'

'History (High School)'

'Social Science (Primary School)'

'Arabic Language (General)'

'Math (Primary School)'

'Arabic Language (Grammar)'

'Economics (High School)'


        
    ]
    
    data = []
    per_subject = max(1, max_samples // len(subjects))
    
    for sub in subjects:
        try:
            ds = load_dataset("MBZUAI/ArabicMMLU", sub, split="test")
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
SYSTEM_PROMPT = ("""أنت وكيل متعدد الأدوار يجمع بين:
1) المفكر الجدلي العميق يفكر باللغة العربية فقط  (مشكك، دقيق، يحلل ويدحض الأخطاء).
2) وكيل التخطيط (يقسم المهام المعقدة إلى خطوات واضحة).
3) وكيل الاستشهاد (يستشهد بالمعلومات المؤكدة فقط، وإلا يصرح بعدم وجود مصدر).

---

⚠️ إرشادات العمل خطوة بخطوة (التزم بها كاملةً):

١. **التحليل والتصنيف**:
حلّل السؤال، صنّفه (بسيطاً أو معقداً)، وحدّد المتغيرات والعلاقات بينها. انظر إلى السؤال من زوايا مختلفة، واختَر التفسير الأكثر منطقية.

٢. **التفكير التكيفي (داخل <|think|>)**:
- إذا كان السؤال **معقداً** (يحوي حسابات، معادلات، أو مفاهيم متشابكة): اكتب تفكيراً داخلياً طويلاً (٥ جمل على الأقل، ويفضّل الزيادة).
- أما إذا كان **بسيطاً**: فاكتفِ بجملتين أو ثلاث.
- **يجب أن يتضمن التفكير**: الشك، المقارنة بين الاحتمالات، دحض الإجابات الخاطئة، وذكر خطأ شائع يقع فيه الناس حول هذا الموضوع.

٣. **طلب المعلومات المفقودة (لمنع الهلوسة)**:
إذا نقصتك معلومة أساسية لحل السؤال، اطلبها مباشرةً من المستخدم ولا تفترضها أبداً.

٤. **التحقق العكسي (إلزامي)**:
بعد أن تصل إلى حل، خصص سطراً واحداً للتحقق العكسي بالصيغة التالية:
"التحقق: لو طبقنا الناتج على المعطيات، هل يتناسب؟ ..."
إذا لم يتناسب الناتج، عد وحاول مجدداً.

٥. **الرد النهائي (داخل <|the_respond|>)**:
- اكتب رداً مباشراً وواضحاً بأسلوب ودود.
- لا تكتفِ بذكر الإجابة الصحيحة؛ بل اشرح باختصار سبب خطأ الخيارات الأخرى.
- إذا كانت الإجابة المرفقة مع السؤال خاطئة علمياً، صحّحها واذكر أن المصدر الأصلي كان خاطئاً.
- كل ردك يجب أن يكون بالعربية الفصحى، ولا تستخدم الإنجليزية إلا للمصطلحات العالمية الضرورية.

٦. **الخلاصة النهائية**:
أنهِ ردك بخلاصة مختصرة جداً (جملة أو جملتين) تحتوي على "الزبدة"، ثم اذكر بوضوح: "الإجابة الصحيحة هي: ...".

---

ابدأ الآن بتنفيذ هذه التعليمات في ردك القادم، والتزم به ١٠٠٪. مثل كانك تفكر قبل أن ترد ثم اكتب ردا بعد التفكير كنماذج DeepSeek-R1 
"""
)

def format_to_micro_engine(user_query, think_content, solve_content):
    return (
        f"<|startoftext|><|system|>{SYSTEM_PROMPT}<|end_context|>\n"
        f"<|startoftext|><|user|>{user_query}<|endoftext|>\n"
        f"<|assistant|><|think|>{think_content}<|end_think|>\n"
        f"<|the_respond|>{solve_content}<|endoftext|>"
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

# ==========================================
# 5. دالة تنقية JSON من الأخطاء (حل Invalid \escape)
# ==========================================
def extract_and_clean_json(raw_text):
    """
    تبحث عن أول JSON صحيح في النص الخام، وتنظف الشرطات المائلة العكسية غير الصالحة
    قبل محاولة تحليلها، خاصةً لحل مشكلة Invalid \escape في النصوص العربية.
    """
    start = raw_text.find('{')
    if start == -1:
        return None

    brace_count = 0
    in_string = False
    end = -1
    
    for i in range(start, len(raw_text)):
        char = raw_text[i]
        if char == '"' and (i == 0 or raw_text[i-1] != '\\'):
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

    if end == -1:
        return None

    json_str = raw_text[start:end]

    # تنظيف الشرطات المائلة غير الصالحة
    def replace_invalid_escape(match):
        char = match.group(1)
        if char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
            return '\\' + char
        elif char == 'u':
            next_chars = json_str[match.end():match.end()+4]
            if re.match(r'^[0-9a-fA-F]{4}$', next_chars):
                return '\\u'
            else:
                return '\\\\u'
        else:
            return '\\\\' + char

    cleaned_json_str = re.sub(r'\\(.)', replace_invalid_escape, json_str)

    try:
        return json.loads(cleaned_json_str, strict=False)
    except json.JSONDecodeError:
        # محاولة أخيرة: استبدال كل \ بـ \\ (حل نووي)
        try:
            fallback_str = json_str.replace('\\', '\\\\')
            fallback_str = fallback_str.replace('\\\\"', '\\"')
            return json.loads(fallback_str, strict=False)
        except:
            return None

# ==========================================
# 6. المعالجة الرئيسية
# ==========================================
def process_samples(samples, output_file):
    if not samples:
        logger.warning("⚠️ لا توجد عينات للمعالجة.")
        return 0, 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    successful = 0
    failed = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(samples, desc="معالجة العينات")):
            # بناء النص (بدون الإجابة الصحيحة)
            if "question" in item:
                user_query = item["question"]
                if "choices" in item:
                    choices_text = "\n".join([f"{chr(65+i)}. {ch}" for i, ch in enumerate(item["choices"])])
                    user_query += f"\nالخيارات:\n{choices_text}"
                    # لا نضيف الإجابة الصحيحة هنا
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
                f"السؤال:\n{user_query}\n\n"
                f"⚠️ استجب بتنسيق JSON حصرياً كالتالي:\n"
                f"{{\n  \"thinking\": \"...\",\n  \"solution\": \"...\"\n}}"
            )

            try:
                raw = call_groq_with_retry(prompt_instruction)
                res = extract_and_clean_json(raw)
                if res is None:
                    raise ValueError("لا يوجد JSON صحيح بعد التنقية.")

                think = res.get("thinking", "").strip()
                solve = res.get("solution", "").strip()
                if not think or not solve:
                    raise ValueError("محتوى فارغ.")

                final_text = format_to_micro_engine(user_query, think, solve)
                record = {
                    "text": final_text,
                    "metadata": {
                        "index": idx,
                        "source": "groq_enriched",
                        "correct_answer": item.get("answer", None),
                        "model_answer": solve
                    }
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                successful += 1
                logger.info(f"✅ تمت معالجة العينة {idx+1}/{len(samples)}")

                time.sleep(REQUEST_DELAY + random.uniform(0, JITTER))

            except Exception as e:
                failed += 1
                logger.error(f"❌ فشل العينة {idx}: {e}")

    return successful, failed

# ==========================================
# 7. عرض الخلاصة النهائية
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
# 8. الدالة الرئيسية
# ==========================================
def main():
    logger.info(f"📂 تحميل عينات من ArabicMMLU (الـ Seed: {RANDOM_SEED})")
    samples = load_arabic_mmlu(max_samples=MAX_SAMPLES, seed=RANDOM_SEED)

    if not samples:
        logger.error("❌ لا توجد بيانات للمعالجة. تأكد من المصادر.")
        sys.exit(1)

    logger.info(f"🚀 بدء المعالجة لـ {len(samples)} عينة...")
    successful, failed = process_samples(samples, OUTPUT_FILE)
    
    print_summary(successful, failed, OUTPUT_FILE)

if __name__ == "__main__":
    main()
