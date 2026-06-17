import os
import json
import time
import logging
from tqdm import tqdm
from groq import Groq

# ==========================================
# 1. الإعدادات العامة وتسجيل المخرجات (Logging)
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ضع مفتاح API الخاص بـ Groq هنا أو في متغيرات البيئة
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"  # أو النموذج الذي تفضله للمهام العامة
GROQ_TEMP = 0.5
GROQ_MAX_TOKENS = 2048

# أسماء الملفات (قم بتعديلها لتطابق أسماء ملفاتك الحقيقية)
INPUT_JSON_FILE = "your_input_data.json"  # اسم ملف الـ JSON الخاص بك هنا 👈
OUTPUT_DATASET_FILE = "micro_engine_train_data.jsonl"

# تهيئة عميل Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
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
# 3. معالجة البيانات واستدعاء الـ API
# ==========================================
def process_and_enrich_dataset(input_file, output_file):
    if not groq_client:
        logger.error("❌ لا يمكن بدء المعالجة بدون عميل API صالح.")
        return

    # أمانة برمجية: التحقق من وجود الملف لتجنب انهيار السكريبت
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

    # إذا كان الملف عبارة عن قائمة من العناصر
    if not isinstance(data_items, list):
        logger.warning("⚠️ تحذير: ملف JSON ليس قائمة (List). سيتم محاولة معالجته كعنصر مفرد.")
        data_items = [data_items]

    logger.info(f"🚀 تم العثور على {len(data_items)} عينة. بدء المعالجة والتوليد عبر Groq...")

    # فتح ملف المخرجات في وضع الإضافة (Append) لضمان الخصائص الاسترجاعية (Checkpointing)
    with open(output_file, "a", encoding="utf-8") as out_f:
        for index, item in enumerate(tqdm(data_items, desc="Processing Items")):
            
            # 🔍 خدعة الاستخراج الزكي: البحث عن مفتاح النص أو السؤال بغض النظر عن اسمه
            # يبحث عن 'question' أو 'text' أو 'prompt' أو 'input'
            user_query = item.get("question") or item.get("text") or item.get("prompt") or item.get("input")
            
            if not user_query:
                logger.warning(f"⚠️ العينة رقم {index} لا تحتوي على مفتاح نصي معروف. تم تخطيها. المفاتيح المتاحة: {list(item.keys())}")
                continue

            # صياغة التعليمات الصارمة لإجبار النموذج الخارجي على الالتزام ببنية الـ JSON المطلوبة
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
            retries = 3
            
            while retries > 0 and not success:
                try:
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a precise data creation engine that outputs strictly JSON."},
                            {"role": "user", "content": prompt_instruction}
                        ],
                        temperature=GROQ_TEMP,
                        max_tokens=GROQ_MAX_TOKENS,
                        response_format={"type": "json_object"}  # إجبار Groq على إخراج JSON نظيف
                    )
                    
                    # تحليل الاستجابة
                    raw_content = response.choices[0].message.content
                    res_json = json.loads(raw_content)
                    
                    think_content = res_json.get("thinking", "").strip()
                    solve_content = res_json.get("solution", "").strip()
                    
                    if not think_content or not solve_content:
                        raise ValueError("محتوى التفكير أو الحل فارغ في استجابة الـ API.")

                    # صياغة النص بالتنسيق الهيكلي النهائي الخاص بـ Zeko Micro-Engine
                    final_sample_text = format_to_micro_engine(user_query, think_content, solve_content)
                    
                    # حفظ العينة بصيغة JSONL متوافقة مع الـ Training Pipeline الخاص بك
                    output_entry = {
                        "text": final_sample_text,
                        "metadata": {
                            "original_index": index,
                            "source_file": input_file
                        }
                    }
                    
                    out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                    out_f.flush()  # كتابة فورية على القرص لضمان عدم ضياع البيانات عند الانقطاع
                    
                    success = True
                    time.sleep(0.5)  # تفادي الـ Rate Limit (RPM) الخاص بـ Groq

                except json.JSONDecodeError:
                    logger.error(f"❌ فشل تحليل الـ JSON من استجابة الـ API في العينة {index}. إعادة المحاولة...")
                    retries -= 1
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"❌ خطأ غير متوقع في العينة {index}: {e}")
                    retries -= 1
                    time.sleep(2)
            
            if not success:
                logger.error(f"🛑 تم تخطي العينة {index} بعد فشل جميع محاولات الاستدعاء.")

    logger.info(f"✨ اكتملت المعالجة! تم حفظ البيانات النهائية المجهزة للتدريب في: {output_file}")

# ==========================================
# 4. نقطة الانطلاق لتشغيل السكريبت
# ==========================================
if __name__ == "__main__":
    # تشغيل الدالة وتمرير أسماء الملفات
    process_and_enrich_dataset(INPUT_JSON_FILE, OUTPUT_DATASET_FILE)
