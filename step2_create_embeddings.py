# -*- coding: utf-8 -*-
"""
الاستخدام:
  1) تأكد من وجود ملف data/qatar_labor_chunks.jsonl.
  2) قم بتثبيت المكتبات اللازمة: pip install sentence-transformers torch tqdm
  3) قم بتشغيل هذا الكود: python step2_create_embeddings.py

سيقوم هذا الكود بإنشاء ملف جديد باسم qatar_labor_embeddings.jsonl في مجلد data.
هذا الملف سيحتوي على نفس بيانات الملف الأصلي، مع إضافة حقل "embedding" لكل مادة قانونية.
"""

import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- الإعدادات الرئيسية ---

# المسار إلى ملف البيانات المصدر
INPUT_PATH = "data/qatar_labor_chunks.jsonl"
# المسار الذي سيتم حفظ الملف الجديد فيه
OUTPUT_PATH = "data/qatar_labor_embeddings.jsonl"

# اسم الموديل المستخدم لتوليد الـ embeddings
# هذا الموديل (bge-m3-law) هو نسخة معدلة من BGE-M3
# وتم تخصيصها للنصوص القانونية، مما يجعلها خياراً ممتازاً لمشروعنا.
MODEL_NAME = 'mhaseeb1604/bge-m3-law'


# --- دوال مساعدة ---

def load_chunks(file_path: str) -> list[dict]:
    """
    تقوم هذه الدالة بقراءة ملف JSONL الذي يحتوي على المواد القانونية.
    """
    if not os.path.exists(file_path):
        print(f"❌ خطأ: لم يتم العثور على الملف {file_path}.")
        print("يرجى التأكد من أن الملف موجود في المسار الصحيح.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_chunks_with_embeddings(file_path: str, data: list[dict]):
    """
    تقوم هذه الدالة بحفظ المواد القانونية مع الـ embeddings الخاصة بها في ملف JSONL جديد.
    """
    # التأكد من وجود مجلد data
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # نستخدم ensure_ascii=False للحفاظ على النصوص العربية بدون ترميز
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# --- الدالة الرئيسية ---

def main():
    """
    الدالة الرئيسية التي تنظم سير عمل البرنامج:
    1. تحميل المواد القانونية من الملف.
    2. تهيئة موديل الـ embeddings.
    3. توليد الـ embeddings للنصوص.
    4. إضافة الـ embeddings إلى البيانات وحفظها في ملف جديد.
    """
    # 1. تحميل البيانات
    print(f"📖 جاري تحميل المواد من ملف: {INPUT_PATH}...")
    chunks = load_chunks(INPUT_PATH)
    if not chunks:
        return
    print(f"✅ تم تحميل {len(chunks)} مادة بنجاح.")

    # 2. تهيئة موديل الـ embeddings
    print(f"🧠 جاري تهيئة موديل الـ embeddings المتخصص: '{MODEL_NAME}'...")
    print("(قد يستغرق هذا بعض الوقت في المرة الأولى لتحميل الموديل)")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ تم تهيئة الموديل بنجاح.")

    # 3. توليد الـ embeddings
    # نستخلص جميع النصوص لمعالجتها دفعة واحدة (batch processing) لزيادة الكفاءة
    texts_to_embed = [chunk['text'] for chunk in chunks]
    print(f"⏳ جاري توليد الـ embeddings لـ {len(texts_to_embed)} نص...")

    # يقوم الموديل بتحويل قائمة النصوص إلى قائمة من الـ embeddings
    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,  # لعرض شريط تقدم
        convert_to_tensor=False  # للحصول على numpy array يمكن تحويله بسهولة إلى قائمة
    )
    print("✅ تم توليد الـ embeddings بنجاح.")

    # 4. إضافة الـ embeddings إلى البيانات وحفظها
    print(f"💾 جاري إضافة الـ embeddings وحفظ الملف الجديد في: {OUTPUT_PATH}...")
    for i, chunk in enumerate(tqdm(chunks, desc="تحديث البيانات")):
        # يجب تحويل الـ embedding إلى قائمة عادية (list) لتكون متوافقة مع صيغة JSON
        chunk['embedding'] = embeddings[i].tolist()

    save_chunks_with_embeddings(OUTPUT_PATH, chunks)

    print("\n--- ✨ اكتملت العملية بنجاح ✨ ---")
    print(f"  - تم إنشاء embeddings لـ {len(chunks)} مادة.")
    print(f"  - تم حفظ الملف الجديد في: {OUTPUT_PATH}")
    print("-------------------------------------\n")
    print(f"🚀 أنت الآن جاهز لتشغيل `step3_load_to_chroma.py` لتحميل البيانات إلى قاعدة البيانات المتجهة.")


if __name__ == "__main__":
    main()