# -*- coding: utf-8 -*-
"""
مساعد قانوني ذكي بواجهة ويب باستخدام Flask
الاستخدام:
  1) تأكد من وجود ملف .env يحتوي على مفتاح Groq API.
  2) تأكد من وجود قاعدة بيانات ChromaDB في مجلد 'chroma_db'.
  3) قم بتشغيل هذا التطبيق: flask --app app run
  4) افتح المتصفح على العنوان: http://127.0.0.1:5000
"""

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from flask import Flask, render_template, request

# --- تحميل وتهيئة الإعدادات والنماذج (يتم مرة واحدة عند بدء التشغيل) ---

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# الإعدادات الرئيسية
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "qatar_labor_law"
EMBEDDING_MODEL_NAME = 'mhaseeb1604/bge-m3-law'
N_RESULTS = 10

# إعدادات Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("لم يتم العثور على مفتاح GROQ_API_KEY في ملف .env. يرجى إضافته.")
GROQ_MODEL_NAME = 'llama3-70b-8192'

# قالب الأوامر (System Prompt) لـ Groq
SYSTEM_PROMPT_TEMPLATE = """
أنت "مساعد قانوني ذكي" متخصص حصراً في قانون العمل القطري. هذه هي مهمتك:

[المدخلات]
- نص القضية: {user_query}
- السياق المسترجع من الـRAG:
{context}

[الهدف]
حلّل نص القضية، واستخرج المواد القانونية ذات الصلة من قانون العمل القطري، وحدّد المخالفات المحتملة والإجراءات/الجزاءات الصحيحة، ثم أعطِ توصيات عملية للطرفين (صاحب العمل/العامل). لا تُقدّم استشارة قانونية رسمية؛ هذا تحليل معلوماتي.

[قواعد صارمة]
1) لا هلوسة: لا تذكر مادة أو فقرة ما لم تكن موجودة في السياق المقدم.
2) رتّب المواد من الأعلى صلة إلى الأقل (بحد أقصى 8 مواد).
3) اربط كل مادة بسبب صِلتها بالوقائع بجملة قصيرة، وأرفق إحالة مرجعية من السياق بصيغة [ref:{{source_id}}].
4) لا تُفصّل "سلسلة تفكيرك"؛ قدّم خلاصات موجزة فقط.
5) إذا وُجد تعارض في النصوص داخل السياق، اذكره باقتضاب وفضّل الأحدث/الأوضح مع تبيين السبب.
6) إن كانت هناك فجوات بيانات تؤثر النتيجة، اذكرها ضمن قسم "نواقص معلومات" دون طرح أسئلة.

[المخرجات — الصيغة المطلوبة]
اكتب بالعربية الفصحى، موجزًا ومنظمًا بالترتيب التالي:

1) **ملخص الحالة:**
   - جملة أو جملتان تلخصان الواقعة الجوهرية.

2) **عناصر واقعية مستخرجة (Facts):**
   - نقاط واضحة: نوع المخالفة، التاريخ أو المدة، الجزاءات المطبقة، وجود مبرر...

3) **المواد ذات الصلة (مرتبة بالأهمية):**
   - **مادة {{article_no}}:** سبب الصلة المختصر (≤ 20 كلمة).
     > "اقتباس قصير جداً من السياق إن لزم" — [ref:{{source_id}}] — (ثقة: {{score}})

4) **المخالفات المحتملة:**
   - نقاط طعن/مخالفة محتملة مرتبطة بأرقام مواد محددة، كل نقطة بجملة سبب.

5) **توصيات عملية:**
   - **لصاحب العمل:** نقطتان إلى ثلاث نقاط سريعة قابلة للتنفيذ.
   - **للعامل:** نقطتان إلى ثلاث نقاط سريعة قابلة للتنفيذ.

6) **تقدير المخاطر:**
   - **على صاحب العمل:** منخفض/متوسط/عالٍ (مع سبب مختصر).
   - **على العامل:** منخفض/متوسط/عالٍ (مع سبب مختصر).

7) **المواد المختصرة (للاقتباس السريع):**
   {{article_no_list_as_inline}}   ← مثال: 61، 62/4، 59/2، 60

8) **تنبيه:**
   هذا تحليل معلوماتي مبني على النص والسياق المقدمين ولا يُعد استشارة قانونية.
"""

print(f"🧠 جاري تحميل نموذج الـ Embedding من: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ تم تحميل نموذج الـ Embedding بنجاح.")

print(f"🤖 جاري تهيئة عميل Groq مع موديل: {GROQ_MODEL_NAME}...")
groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ تم تهيئة Groq بنجاح.")

print("\n🧠 جاري الاتصال بقاعدة بيانات ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
distance_metric = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"✅ قاعدة البيانات جاهزة. (مقياس المسافة المستخدم: {distance_metric})")
if distance_metric != "cosine":
    print("⚠️ تحذير: قاعدة البيانات لا تستخدم 'cosine'. قد تكون النتائج غير دقيقة.")

# --- تهيئة تطبيق Flask ---
app = Flask(__name__)

# --- الدوال المساعدة ---
def format_context_for_llm(results: dict, distance_metric: str) -> str:
    """تنسيق نتائج البحث من ChromaDB إلى نص مفهوم للنموذج اللغوي."""
    context_parts = []
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    ids = results['ids'][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        similarity = 1 - dist if distance_metric == "cosine" else (1 - (dist ** 2) / 2)
        source_id = ids[i]
        article_no = meta.get('article_label', 'N/A')
        context_item = (
            f"عنصر سياق رقم {i+1}:\n"
            f"- source_id: {source_id}\n"
            f"- article_no: {article_no}\n"
            f"- text: \"{doc.strip()}\"\n"
            f"- score: {similarity:.4f}\n"
        )
        context_parts.append(context_item)
    return "\n---\n".join(context_parts)

def get_groq_analysis(user_query: str, context: str) -> str:
    """إرسال السؤال والسياق إلى Groq للحصول على تحليل قانوني."""
    try:
        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_query=user_query, context=context)
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model=GROQ_MODEL_NAME,
            temperature=0.2,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ حدث خطأ أثناء الاتصال بـ Groq: {e}"

# --- مسارات (Routes) تطبيق الويب ---

@app.route('/')
def index():
    """عرض الصفحة الرئيسية التي تحتوي على نموذج الإدخال."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """معالجة طلب التحليل من المستخدم."""
    user_query = request.form['query']
    if not user_query:
        return render_template('index.html', error="الرجاء إدخال نص القضية.")

    # 1. استرجاع المواد ذات الصلة (RAG - Retrieval)
    query_embedding = embedding_model.encode(user_query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
        include=['metadatas', 'documents', 'distances']
    )

    if not results['documents'][0]:
        analysis_result = "لم يتم العثور على مواد قانونية ذات صلة بالحالة الموصوفة."
    else:
        # 2. تنسيق السياق وإرساله للتحليل (Generation)
        context_str = format_context_for_llm(results, distance_metric)
        analysis_result = get_groq_analysis(user_query, context_str)

    # 3. عرض صفحة النتائج
    # استبدال علامات السطر الجديد بـ <br> لعرضها بشكل صحيح في HTML
    formatted_analysis = analysis_result.replace('\n', '<br>')
    return render_template('results.html', query=user_query, analysis=formatted_analysis)

if __name__ == '__main__':
    # لتشغيل التطبيق، استخدم الأمر: flask --app app run
    # أو قم بتشغيله كأي ملف بايثون عادي
    app.run(debug=True)
