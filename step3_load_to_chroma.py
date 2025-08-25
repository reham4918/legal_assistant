# -*- coding: utf-8 -*-
"""
الاستخدام:
  1) تأكد من تشغيل step2_create_embeddings.py أولاً لإنتاج qatar_labor_embeddings.jsonl.
  2) قم بتثبيت المكتبات اللازمة: pip install chromadb tqdm
  3) قم بتشغيل هذا الكود: python step3_load_to_chroma.py

سينتج هذا الكود قاعدة بيانات متجهة (Vector DB) على القرص في مجلد chroma_db،
ويقوم بتخزين وفهرسة جميع المواد القانونية مع الـ embeddings الخاصة بها.
"""

import json
import os
import chromadb
from tqdm import tqdm

# --- الإعدادات الرئيسية ---

# المسار إلى ملف البيانات مع الـ embeddings
INPUT_PATH = "data/qatar_labor_embeddings.jsonl"
# المسار الذي سيتم حفظ قاعدة البيانات فيه
CHROMA_PATH = "chroma_db"
# اسم المجموعة (collection) داخل قاعدة البيانات
COLLECTION_NAME = "qatar_labor_law"


# --- دوال مساعدة ---

def load_embeddings_data(file_path: str) -> list[dict]:
    """
    تقوم هذه الدالة بقراءة ملف JSONL الذي يحتوي على الـ embeddings.
    """
    if not os.path.exists(file_path):
        print(f"❌ خطأ: لم يتم العثور على الملف {file_path}.")
        print("يرجى التأكد من تشغيل `step2_create_embeddings.py` أولاً.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# --- الدالة الرئيسية ---

def main():
    """
    الدالة الرئيسية التي تنظم سير عمل البرنامج:
    1. تحميل البيانات مع الـ embeddings.
    2. تهيئة ChromaDB client.
    3. إنشاء أو الحصول على collection.
    4. إضافة البيانات إلى الـ collection.
    """
    # 1. تحميل البيانات
    print(f"📖 جاري تحميل المواد والـ embeddings من ملف: {INPUT_PATH}...")
    chunks = load_embeddings_data(INPUT_PATH)
    if not chunks:
        return
    print(f"✅ تم تحميل {len(chunks)} مادة بنجاح.")

    # 2. تهيئة ChromaDB client
    # نستخدم PersistentClient لحفظ قاعدة البيانات على القرص الصلب في المسار المحدد
    print(f"🧠 جاري تهيئة قاعدة بيانات ChromaDB في المسار: '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 3. إنشاء أو الحصول على collection
    # get_or_create_collection تضمن عدم حدوث خطأ لو تم تشغيل الكود أكثر من مرة
    print(f"🔍 جاري إنشاء أو الحصول على المجموعة (collection) باسم: '{COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("✅ تم الحصول على المجموعة بنجاح.")

    # 4. إضافة البيانات إلى الـ collection
    # نقوم بتجهيز البيانات بالشكل الذي تتوقعه مكتبة ChromaDB
    # وهي قوائم منفصلة لكل من: ids, documents, metadatas, embeddings
    print(f"➕ جاري إضافة {len(chunks)} مادة إلى قاعدة البيانات...")

    # استخلاص البيانات في قوائم منفصلة
    ids = [chunk['chunk_id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    embeddings = [chunk['embedding'] for chunk in chunks]

    # إضافة البيانات دفعة واحدة (لأن عددها قليل)
    # في حالة البيانات الضخمة، يفضل تقسيمها إلى دفعات (batches)
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # طباعة ملخص للعملية
    count = collection.count()
    print("\n--- ✨ اكتملت العملية بنجاح ✨ ---")
    print(f"  - تم إضافة وفهرسة {count} مادة في قاعدة البيانات.")
    print(f"  - اسم المجموعة: {COLLECTION_NAME}")
    print(f"  - تم حفظ قاعدة البيانات في المجلد: {CHROMA_PATH}")
    print("-------------------------------------\n")
    print("🚀 أنت الآن جاهز لبناء نظام الاسترجاع (RAG) والبدء في طرح الأسئلة!")


if __name__ == "__main__":
    main()