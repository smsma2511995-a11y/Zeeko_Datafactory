#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
مصنع البيانات الذكي لـ "زيكو" – إصدار يعتمد على Gemini API (google-genai) فقط
لتوليد بيانات عربية دقيقة وخالية من الأخطاء الإملائية (خاصة التمييز بين الياء والتنوين).
"""
import os
import json
import logging
import time
import re
import yaml
import subprocess
import sys
from tqdm import tqdm
from datasets import load_dataset

# 1. تأمين الاستيراد بالطريقة الصحيحة للمكتبة الجديدة
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    print("✅ تم العثور على مكتبة Gemini بنجاح")
except ImportError:
    print("🔄 المكتبة مش مقروءة.. جاري محاولة الربط المباشر...")
    try:
        # محاولة إجبار بايثون على رؤية المكتبات المثبتة
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai", "json5", "pyyaml"])
        from google import genai
        from google.genai import types
        GENAI_AVAILABLE = True
    except:
        GENAI_AVAILABLE = False
        print("❌ فشل نهائي في العثور على google-genai")
        exit(1)

# 2. تأمين json5
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    JSON5_AVAILABLE = False

# -------------------- الإعدادات --------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

gemini_cfg = config.get("gemini", {})

# إعدادات المعالجة
TEMPERATURE = 0.3
MAX_NEW_TOKENS_DEFAULT = 512
MAX_NEW_TOKENS_CODING = 2048   # للمهام البرمجية (إذا أردنا استخدام Gemini لها)
GEMINI_DELAY = 10               # تأخير بين الطلبات لتجنب حدود المعدل
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# ملفات الإخراج
OUTPUT_FILE = "data/enriched_training_data.jsonl"
FAILED_OUTPUT_FILE = "data/failed_samples.jsonl"
os.makedirs("data", exist_ok=True)

# -------------------- إعداد Gemini (باستخدام google.genai.Client) --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or gemini_cfg.get("api_key")
if not GEMINI_API_KEY:
    logger.error("❌ لم يتم العثور على مفتاح Gemini. الرجاء تعيين GEMINI_API_KEY في البيئة أو config.yaml")
    exit(1)

# إنشاء العميل
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={'api_version': 'v1'}  # مهم لاستقرار الاتصال
)
GEMINI_MODEL = gemini_cfg.get("model", "gemini-2.0-flash-lite")  # أو أي نموذج تريده
logger.info(f"✅ تم تهيئة Gemini باستخدام الموديل {GEMINI_MODEL}")

# -------------------- دوال مساعدة --------------------
def extract_json(text):
    """استخراج JSON من النص مع محاولات إصلاح متعددة."""
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if not match:
        return {}, None
    json_str = match.group(1)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    def try_parse(s):
        try:
            return json.loads(s), s
        except json.JSONDecodeError:
            try:
                s_no_comments = re.sub(r'//.*?\n', '', s)
                return json.loads(s_no_comments), s_no_comments
            except:
                if JSON5_AVAILABLE:
                    try:
                        return json5.loads(s), s
                    except:
                        pass
                s_fixed = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', s)
                try:
                    return json.loads(s_fixed), s_fixed
                except:
                    return {}, None
    return try_parse(json_str)

def check_logical_consistency(item, data):
    """فحص بسيط للاتساق المنطقي (اختياري)."""
    task = item.get("task")
    if task == "mcq":
        question = item.get("question", "")
        keywords = set(re.findall(r'\b\w{4,}\b', question))
        if not keywords:
            return True
        explain = data.get("explain_concept", "")
        if len(explain) < 20 or not any(kw in explain for kw in keywords):
            logger.debug(f"فحص منطقي فاشل: {keywords} vs {explain[:50]}")
            return False
    return True

# -------------------- تحميل المصادر (كما هي بدون تغيير) --------------------
def load_arabic_mmlu(split="test", max_samples=50):
    logger.info("جاري تحميل ArabicMMLU...")
    subjects = ['Physics (High School)', 'Biology (High School)', 'Arabic Language (High School)', 'Arabic Language (Grammar)']
    data = []
    for sub in subjects:
        try:
            ds = load_dataset("MBZUAI/ArabicMMLU", sub, split=split, streaming=True)
            count = 0
            for item in ds:
                if max_samples and count >= (max_samples // len(subjects)):
                    break
                question = item.get("Question")
                options = [
                    item.get("Option 1"),
                    item.get("Option 2"),
                    item.get("Option 3"),
                    item.get("Option 4")
                ]
                options = [opt for opt in options if opt and str(opt).strip()]
                answer_key = item.get("Answer Key")
                if not question or not options or len(options) < 2 or not answer_key:
                    continue
                data.append({
                    "source": f"MMLU_{sub}",
                    "subject": sub,
                    "question": question,
                    "choices": options,
                    "answer": answer_key,
                    "task": "mcq"
                })
                count += 1
            logger.info(f"✅ تم تحميل {count} عينة من {sub}.")
        except Exception as e:
            logger.warning(f"❌ مادة {sub} تخطيت بسبب خطأ: {e}")
    return data

def load_aya_arabic(max_samples=2000):
    target_langs = ["standard_arabic", "egyptian_arabic", "french", "english"]
    samples_per_lang = max_samples // len(target_langs)
    mixed_data = []
    for lang in target_langs:
        try:
            logger.info(f"جاري سحب بيانات: {lang}")
            ds = load_dataset("CohereLabs/aya_collection_language_split", lang, split="train", streaming=True)
            count = 0
            for item in ds:
                if count >= samples_per_lang:
                    break
                if not item.get("inputs"):
                    continue
                mixed_data.append({
                    "instruction": item["inputs"],
                    "output": item["targets"],
                    "language": lang,
                    "task": "general_qa"
                })
                count += 1
            logger.info(f"✅ تم تحميل {count} عينة من Aya للغة {lang}.")
        except Exception as e:
            logger.error(f"❌ فشل سحب {lang}: {e}")
    return mixed_data

def load_squad_arabic(max_samples=None):
    logger.info("جاري تحميل SQuAD Arabic...")
    try:
        ds = load_dataset("Mostafa3zazi/Arabic_SQuAD", split="train", streaming=True)
        data = []
        count = 0
        for item in ds:
            if max_samples and count >= max_samples:
                break
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", {})
            if context and question and answers:
                data.append({
                    "source": "Arabic_SQuAD",
                    "context": context,
                    "question": question,
                    "answers": answers,
                    "task": "reading"
                })
                count += 1
        logger.info(f"✅ تم تحميل {len(data)} عينة من Arabic_SQuAD.")
        return data
    except Exception as e:
        logger.error(f"❌ فشل تحميل SQuAD العربي: {e}")
        return []

def load_french_mmlu(max_samples=500):
    logger.info("جاري تحميل French MMLU...")
    try:
        ds = load_dataset("FreedomIntelligence/MMLU_French", split="train", streaming=True)
        data = []
        for count, item in enumerate(ds):
            if max_samples and count >= max_samples:
                break
            question = item.get("Question") or item.get("question")
            choices = []
            for letter in ['A', 'B', 'C', 'D']:
                if item.get(letter):
                    choices.append(item.get(letter))
            answer = item.get("Answer") or item.get("answer")
            subject = item.get("Subject", "General French")
            if not question or not choices or not answer:
                continue
            data.append({
                "source": "French_MMLU_Custom",
                "subject": subject,
                "question": question,
                "choices": choices,
                "answer": answer,
                "task": "mcq"
            })
        logger.info(f"✅ تم تحميل {len(data)} عينة من French MMLU.")
        return data
    except Exception as e:
        logger.error(f"❌ فشل تحميل French MMLU: {e}")
        return []

def load_code_contests(max_samples=500):
    logger.info(f"📡 جاري سحب {max_samples} عينة برمجة...")
    try:
        ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
        data = []
        for item in ds.take(max_samples):
            if item.get("solutions") and len(item["solutions"]["solution"]) > 0:
                data.append({
                    "source": "CodeContests_Small",
                    "problem_description": item["description"],
                    "code_solution": item["solutions"]["solution"][0],
                    "task": "coding"
                })
        logger.info(f"✅ تم تجهيز جرعة البرمجة: {len(data)} عينة.")
        return data
    except Exception as e:
        logger.error(f"❌ خطأ في سحب الكود: {e}")
        return []

def load_xlsum_arabic(split="train", max_samples=None):
    logger.info("جاري تحميل XLSum Arabic...")
    try:
        ds = load_dataset("csebuetnlp/xlsum", "arabic", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        data = []
        for item in ds:
            if not item.get("text"):
                continue
            data.append({
                "source": "XLSum",
                "article": item["text"],
                "summary": item["summary"],
                "task": "summary"
            })
        logger.info(f"✅ تم تحميل {len(data)} عينة من XLSum.")
        return data
    except Exception as e:
        logger.error(f"❌ فشل تحميل XLSum: {e}")
        return []

def load_english_mmlu(max_samples=500):
    logger.info("جاري تحميل English MMLU...")
    subjects = [
        "college_mathematics", "high_school_physics", "high_school_chemistry",
        "high_school_biology", "high_school_computer_science",
        "high_school_macroeconomics", "high_school_psychology", "professional_law"
    ]
    data = []
    samples_per_subject = max(1, max_samples // len(subjects)) if max_samples else 10
    for sub in subjects:
        try:
            ds = load_dataset("cais/mmlu", sub, split="dev", streaming=True)
            count = 0
            for item in ds:
                if count >= samples_per_subject:
                    break
                if not item.get("question"):
                    continue
                data.append({
                    "source": f"English_MMLU_{sub}",
                    "subject": sub,
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "task": "mcq"
                })
                count += 1
            logger.info(f"✅ تم تحميل {count} سؤال من {sub}")
        except Exception as e:
            logger.warning(f"⚠️ فشل مادة {sub}: {e}")
    return data

def load_squad_english(max_samples=None):
    logger.info("جاري تحميل SQuAD English...")
    try:
        ds = load_dataset("squad", split="train", streaming=True)
        data = []
        count = 0
        for item in ds:
            if max_samples and count >= max_samples:
                break
            if not item.get("context") or not item.get("question"):
                continue
            data.append({
                "source": "SQuAD_EN",
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"],
                "task": "reading"
            })
            count += 1
        logger.info(f"✅ تم تحميل {len(data)} عينة من SQuAD English.")
        return data
    except Exception as e:
        logger.error(f"❌ فشل تحميل SQuAD English: {e}")
        return []

def load_gsm8k_arabic(max_samples=50):
    logger.info("جاري تحميل GSM8K لإنشاء مسائل عربية...")
    try:
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        data = []
        count = 0
        for item in ds:
            if max_samples and count >= max_samples:
                break
            if not item.get("question"):
                continue
            data.append({
                "source": "GSM8K",
                "question_en": item["question"],
                "answer_en": item["answer"],
                "task": "math"
            })
            count += 1
        logger.info(f"✅ تم تحميل {len(data)} مسألة من GSM8K.")
        return data
    except Exception as e:
        logger.error(f"❌ فشل تحميل GSM8K: {e}")
        return []

# -------------------- بناء النصوص (Prompts) --------------------
def build_prompt(item):
    task = item.get("task")
    identity = (
        "أنت 'زيكو'، مساعد تعليمي ذكي مصري. يجب أن تكون جميع إجاباتك "
        "باللغة العربية الفصحى المبسطة (مع لمسة مصرية محببة). "
        "لا تستخدم أي كلمة إنجليزية أو لغة أخرى نهائياً."
    )
    json_instruction = "\n\nاكتب JSON صالحاً فقط (بدون أي شرح إضافي). تأكد من أن المحتوى داخل JSON بالعربية فقط."

    if task == "mcq":
        subject = item.get("subject", "غير معروف")
        question = item["question"]
        choices = item["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            return None
        choices_text = "\n".join([f"{chr(65+i)}. {ch}" for i, ch in enumerate(choices)])
        correct = item["answer"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "لفهم السؤال، نلاحظ أن الكميات المشتقة هي تلك التي تعتمد على كميات أساسية. السرعة مثلاً تعتمد على الطول والزمن.",
    "explain_concept": "الكميات المشتقة في الفيزياء هي كميات يتم اشتقاقها من الكميات الأساسية (الطول، الكتلة، الزمن، ...). مثل: السرعة (طول/زمن)، القوة (كتلة * تسارع).",
    "simplify_child": "تخيل أن لديك سيارة لعبة. إذا تحركت مسافة طويلة في زمن قصير، نقول إن سرعتها عالية. السرعة هنا هي كمية مشتقة من المسافة والزمن.",
    "generate_mcq": "أي من التالي يعتبر كمية مشتقة؟\\nأ. الطول\\nب. الكتلة\\nج. السرعة\\nد. الزمن\\nالإجابة: ج"
}
"""
        return f"""{identity}

المهمة: تحليل سؤال متعدد الخيارات في مادة {subject}.

{example}

الآن قم بتحليل السؤال التالي بنفس التنسيق:

السؤال: {question}
الخيارات:
{choices_text}
الإجابة الصحيحة: {correct}

المطلوب: JSON يحتوي على الحقول الأربعة كما في المثال.
{json_instruction}"""

    elif task == "math":
        question_en = item["question_en"]
        answer_en = item["answer_en"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "المسألة تتطلب حساب السرعة المتوسطة. نستخدم القانون: السرعة = المسافة / الزمن.",
    "step_by_step": "أولاً: نحدد المسافة الكلية = 120 كم. ثانياً: الزمن الكلي = 2 ساعة. ثالثاً: نقسم المسافة على الزمن: 120 / 2 = 60 كم/س.",
    "solve_math": "60",
    "reminding": "تذكر أن وحدة السرعة تعتمد على وحدات المسافة والزمن. إذا كانت المسافة بالكيلومتر والزمن بالساعة، فالسرعة تكون كم/س."
}
"""
        return f"""{identity}

المهمة: حل مسألة رياضية.

{example}

المسألة (بالإنجليزية): {question_en}
الحل المرجعي: {answer_en}

المطلوب: JSON يحتوي على الحقول الأربعة كما في المثال.
{json_instruction}"""

    elif task == "reading":
        context = item["context"]
        question = item["question"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "أبحث في النص عن جملة تتحدث عن سبب هجرة الطيور. أجد في الفقرة الثانية: 'تهاجر الطيور هرباً من البرد'.",
    "explain": "النص يوضح أن الطيور تهاجر في الشتاء بسبب انخفاض درجات الحرارة ونقص الغذاء.",
    "summarize_ar": "تتحدث الفقرة عن هجرة الطيور في الشتاء وأسبابها.",
    "generate_quiz": "ما السبب الرئيسي لهجرة الطيور؟\\nالإجابة: الهرب من البرد والبحث عن غذاء."
}
"""
        return f"""{identity}

المهمة: تحليل نص والإجابة عن سؤال.

{example}

النص: {context}
السؤال: {question}

المطلوب: JSON يحتوي على الحقول الأربعة كما في المثال.
{json_instruction}"""

    elif task == "coding":
        desc = item["problem_description"]
        code = item["code_solution"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "المشكلة تطلب إيجاد أكبر عدد في مصفوفة. يمكن حلها بالمرور على جميع العناصر وتحديث القيمة العظمى.",
    "explain_concept": "خوارزمية البحث عن القيمة العظمى تعتمد على مقارنة كل عنصر بقيمة حالية وتحديثها إذا كان العنصر أكبر.",
    "review_code": "الكود الحالي يعمل لكن يمكن تحسينه باستخدام دالة max() الجاهزة في بايثون.",
    "write_code": "مسألة جديدة: اكتب دالة ترجع أصغر عنصر في مصفوفة.\\n```python\\ndef find_min(arr):\\n    min_val = arr[0]\\n    for num in arr:\\n        if num < min_val:\\n            min_val = num\\n    return min_val\\n```"
}
"""
        return f"""{identity}

المهمة: تحليل مشكلة برمجية وحلها.

{example}

وصف المشكلة: {desc}
الكود: {code}

المطلوب: JSON يحتوي على الحقول الأربعة كما في المثال.
{json_instruction}"""

    elif task == "general_qa":
        instruction = item["instruction"]
        original_response = item["output"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "الإجابة الأصلية صحيحة لكنها مختصرة. يمكن إضافة المزيد من الأمثلة لتوضيح الفكرة.",
    "explain": "سأضيف شرحاً عن استخدامات الفعل في الحياة اليومية مع أمثلة.",
    "enhanced_response": "الفعل المضارع يعبر عن حدث يحدث الآن أو في المستقبل. مثال: يكتب الطالب الدرس. يمكن استخدامه أيضاً مع (سوف) للدلالة على المستقبل."
}
"""
        return f"""{identity}

المهمة: تحسين إجابة لتعليمة.

{example}

التعليمة: {instruction}
الإجابة الأصلية: {original_response}

المطلوب: JSON يحتوي على الحقول الثلاثة كما في المثال.
{json_instruction}"""

    elif task == "summary":
        article = item["article"][:1500]
        summary = item["summary"]

        example = """
مثال على الإجابة المطلوبة:
{
    "thought": "الملخص يغطي النقاط الرئيسية، يمكن استخراج أسئلة فهم عميق منه.",
    "questions": [
        {"q": "ما هي الفكرة الرئيسية في المقال؟", "a": "تأثير التكنولوجيا على التعليم."},
        {"q": "اذكر مثالاً على تطبيق تكنولوجي في الفصول.", "a": "السبورات الذكية."},
        {"q": "كيف يمكن للتكنولوجيا تحسين تجربة الطالب؟", "a": "بتوفير محتوى تفاعلي."}
    ],
    "lesson_summary": "التكنولوجيا تساعد في جعل التعليم أكثر تفاعلية وفعالية."
}
"""
        return f"""{identity}

المهمة: استخراج دروس من مقال.

{example}

المقال: {article}
الملخص: {summary}

المطلوب: JSON يحتوي على الحقول الثلاثة كما في المثال.
{json_instruction}"""

    else:
        logger.warning(f"مهمة غير معروفة: {task}")
        return None

def parse_response(item, response):
    data, _ = extract_json(response)
    if not data:
        return response, {}
    if not check_logical_consistency(item, data):
        return response, {}

    task = item.get("task")
    full_response = ""

    if task == "mcq":
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|explain_concept|>\n{data.get('explain_concept', '')}\n\n"
            f"<|simplify_for_child|>\n{data.get('simplify_child', '')}\n\n"
            f"<|generate_mcq|>\n{data.get('generate_mcq', '')}"
        )
    elif task == "math":
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|step_by_step|>\n{data.get('step_by_step', '')}\n\n"
            f"<|solve_math|>\n{data.get('solve_math', '')}\n\n"
            f"<|reminding|>\n{data.get('reminding', '')}"
        )
    elif task == "reading":
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|explain|>\n{data.get('explain', '')}\n\n"
            f"<|summarize_ar|>\n{data.get('summarize_ar', '')}\n\n"
            f"<|generate_quiz|>\n{data.get('generate_quiz', '')}"
        )
    elif task == "coding":
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|explain_concept|>\n{data.get('explain_concept', '')}\n\n"
            f"<|review_code|>\n{data.get('review_code', '')}\n\n"
            f"<|write_code|>\n{data.get('write_code', '')}"
        )
    elif task == "general_qa":
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|explain|>\n{data.get('explain', '')}\n\n"
            f"<|follow_instruction|>\n{data.get('enhanced_response', '')}"
        )
    elif task == "summary":
        questions_text = ""
        for q in data.get('questions', []):
            questions_text += f"سؤال: {q.get('q', '')}\nإجابة: {q.get('a', '')}\n\n"
        full_response = (
            f"<|think|>\n{data.get('thought', '')}\n<|end_think|>\n\n"
            f"<|lesson_summary|>\n{data.get('lesson_summary', '')}\n\n"
            f"<|generate_quiz|>\n{questions_text.strip()}"
        )

    return full_response.strip(), data

def validate_item(it):
    """التحقق من صحة العينة بناءً على نوع المهمة."""
    task = it.get("task")
    if task == "mcq" or task == "reading":
        return bool(it.get("question") and str(it.get("question")).strip())
    elif task == "math":
        return bool(it.get("question_en") and str(it.get("question_en")).strip())
    elif task == "coding":
        return bool(it.get("problem_description") and str(it.get("problem_description")).strip())
    elif task == "general_qa":
        return bool(it.get("instruction") and str(it.get("instruction")).strip())
    elif task == "summary":
        return bool(it.get("article") and str(it.get("article")).strip())
    else:
        return True

# -------------------- معالجة باستخدام Gemini (باستخدام client.models.generate_content) --------------------
def process_all_items_with_gemini(items, output_file):
    if not items:
        return

    # استئناف العمل: حساب عدد العينات المعالجة مسبقاً
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
    logger.info(f"عدد العينات الموجودة مسبقاً في {output_file}: {processed_count}")

    start_index = processed_count
    if start_index >= len(items):
        logger.info("✅ تمت معالجة جميع العينات مسبقاً.")
        return

    items = items[start_index:]
    logger.info(f"بدء معالجة Gemini من العينة {start_index+1} إلى {len(items)+start_index}")

    with open(output_file, "a", encoding="utf-8") as f_out, \
         open(FAILED_OUTPUT_FILE, "a", encoding="utf-8") as f_fail, \
         tqdm(total=len(items), desc="🤖 معالجة Gemini", unit="عينة") as pbar:

        for idx, it in enumerate(items):
            success = False
            attempts = 0
            max_attempts = RETRY_ATTEMPTS

            while not success and attempts < max_attempts:
                try:
                    prompt = build_prompt(it)
                    if not prompt:
                        logger.warning(f"فشل بناء prompt للعينة {start_index+idx}")
                        f_fail.write(json.dumps({"item": it, "error": "فشل بناء prompt"}, ensure_ascii=False) + "\n")
                        f_fail.flush()
                        break

                    # اختيار عدد الرموز حسب نوع المهمة
                    max_tokens = MAX_NEW_TOKENS_CODING if it.get("task") == "coding" else MAX_NEW_TOKENS_DEFAULT

                    # استدعاء Gemini API باستخدام client.models.generate_content
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            max_output_tokens=max_tokens,
                        )
                    )
                    reply = response.text

                    # تحليل الرد
                    full_response, metadata = parse_response(it, reply)
                    if not metadata:
                        # فشل في استخراج JSON أو الفحص المنطقي
                        f_fail.write(json.dumps({"item": it, "response": reply}, ensure_ascii=False) + "\n")
                        f_fail.flush()
                        success = True  # نعتبره فشل ولن نعيد المحاولة
                        break

                    # تسجيل النتيجة الناجحة
                    record = {
                        "source": it.get("source", "unknown"),
                        "task": it["task"],
                        "output": full_response,
                        "metadata": metadata
                    }
                    for k, v in it.items():
                        if k not in record and k not in ["task"]:
                            record[f"original_{k}"] = v

                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    success = True

                except exceptions.ResourceExhausted as e:
                    attempts += 1
                    wait_time = 60  # انتظار دقيقة كاملة عند نفاد الحصة
                    logger.warning(f"⚠️ الحصة نفدت! سأنتظر {wait_time} ثانية (محاولة {attempts}/{max_attempts})...")
                    time.sleep(wait_time)
                except Exception as e:
                    attempts += 1
                    logger.error(f"❌ خطأ غير متوقع: {e} (محاولة {attempts}/{max_attempts})")
                    if attempts < max_attempts:
                        time.sleep(RETRY_DELAY * (2 ** attempts))
                    else:
                        f_fail.write(json.dumps({"item": it, "error": str(e)}, ensure_ascii=False) + "\n")
                        f_fail.flush()
                        break

            # تأخير بين الطلبات الناجحة (حتى لو فشلت بعض المحاولات)
            time.sleep(GEMINI_DELAY)
            pbar.update(1)

# -------------------- الدالة الرئيسية --------------------
def main():
    max_samples = config.get("data", {}).get("max_samples", 500)

    # تجميع البيانات من جميع المصادر
    all_data = []
    all_data.extend(load_arabic_mmlu(max_samples=max_samples))
    all_data.extend(load_aya_arabic(max_samples=max_samples))
    all_data.extend(load_squad_arabic(max_samples=max_samples))
    all_data.extend(load_french_mmlu(max_samples=max_samples))
    all_data.extend(load_code_contests(max_samples=max_samples))
    all_data.extend(load_xlsum_arabic(max_samples=max_samples))
    all_data.extend(load_english_mmlu(max_samples=max_samples))
    all_data.extend(load_squad_english(max_samples=max_samples))
    all_data.extend(load_gsm8k_arabic(max_samples=max_samples))

    logger.info(f"🚀 إجمالي العينات المجمعة قبل التنقية: {len(all_data)}")

    # فلترة العينات غير الصالحة
    all_data = [it for it in all_data if validate_item(it)]
    logger.info(f"✅ العينات الصالحة بعد حذف الفارغ: {len(all_data)}")

    if not all_data:
        logger.error("❌ لا توجد بيانات صالحة!")
        return

    # معالجة جميع العينات عبر Gemini
    process_all_items_with_gemini(all_data, OUTPUT_FILE)

    logger.info(f"✅ تم الانتهاء من المعالجة المتاحة في {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
