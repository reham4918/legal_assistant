# -*- coding: utf-8 -*-
"""
Usage:
  1) Install libraries: pip install sentence-transformers chromadb python-dotenv
  2) Run the code: python step4_ask.py
  3) This script retrieves the top 5 most relevant articles and shows their similarity score.
"""

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- Load environment variables from .env file ---
load_dotenv()

# --- Main Settings ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "qatar_labor_law"
EMBEDDING_MODEL_NAME = 'mhaseeb1604/bge-m3-law'
N_RESULTS = 5

# --- Initialize Embedding Model ---
print(f"🧠 جاري تحميل نموذج الـ Embedding من: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ تم تحميل نموذج الـ Embedding بنجاح.")


def main():
    """
    Main function for retrieving the top N relevant legal articles and their similarity scores.
    """
    print("\n🧠 جاري الاتصال بقاعدة بيانات ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    distance_metric = (collection.metadata or {}).get("hnsw:space", "l2")
    print(f"✅ قاعدة البيانات جاهزة. (مقياس المسافة المستخدم: {distance_metric})")

    if distance_metric == "cosine":
        print("✅ معلومة: قاعدة البيانات تستخدم مقياس Cosine، وهو الموصى به.")
    else:
        print("⚠️ تحذير: قاعدة البيانات تستخدم مقياس L2. للحصول على أفضل النتائج، يوصى بإعادة إنشائها باستخدام 'cosine'.")

    print(f"\n--- ⚖️ أداة استرجاع أفضل {N_RESULTS} مواد قانونية ⚖️ ---")
    print("ادخل وصف الحالة، أو اكتب 'خروج' لإنهاء البرنامج.")
    print("----------------------------------------------------------\n")

    while True:
        user_input = input("📄 ادخل وصف الحالة للبحث عن المواد المتعلقة بها: ")
        if user_input.lower() in ['خروج', 'exit', 'quit']:
            print("👋 مع السلامة!")
            break
        if not user_input:
            continue

        print(f"\n🔍 جاري البحث عن أفضل {N_RESULTS} مواد ذات صلة...")
        query_embedding = embedding_model.encode(user_input, convert_to_tensor=False).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances']
        )

        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        retrieved_distances = results['distances'][0]

        print("\n--- 📜 أفضل المواد القانونية التي تم العثور عليها ---")
        if not retrieved_docs:
            print("   لم يتم العثور على مواد ذات صلة بالحالة الموصوفة.")
        else:
            processed_results = []
            for doc, meta, dist in zip(retrieved_docs, retrieved_metadatas, retrieved_distances):
                similarity = 1 - dist if distance_metric == "cosine" else 1 - (dist ** 2 / 2)
                processed_results.append({
                    "doc": doc,
                    "meta": meta,
                    "similarity": similarity
                })

            print(f"   تم العثور على {len(processed_results)} مواد ذات صلة. ملخص النتائج:\n")
            for i, result in enumerate(processed_results):
                article_label = result['meta'].get('article_label', 'مادة غير محددة')
                similarity_score = result['similarity']
                print(f"  {i+1}. {article_label} (درجة التشابه: {similarity_score:.2%})")

            print("\n--- محتوى المواد بالتفصيل ---\n")
            for i, result in enumerate(processed_results):
                article_label = result['meta'].get('article_label', 'مادة غير محددة')
                chapter_title = result['meta'].get('chapter_title', 'فصل غير محدد')
                similarity_score = result['similarity']
                doc_content = result['doc']

                print(f"📄 المادة رقم {i + 1}: {article_label} (من فصل: {chapter_title})")
                print(f"   ✨ درجة التشابه: {similarity_score:.2%}")
                print("=" * 40)
                print(doc_content.strip())
                print("-" * 40 + "\n")


if __name__ == "__main__":
    main()