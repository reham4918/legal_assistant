# -*- coding: utf-8 -*-
"""
assistant_core.py

هذا الملف يحتوي على المنطق الأساسي للمساعد القانوني.
تم فصل هذا المنطق في كلاس واحد (`LegalAssistant`) لتجنب تكرار الكود
وتسهيل الصيانة والتطوير لكل من واجهة سطر الأوامر (CLI) وواجهة الويب (Flask).
"""

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- تحميل متغيرات البيئة من ملف .env ---
load_dotenv()

# --- الإعدادات الرئيسية ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "qatar_labor_law"
EMBEDDING_MODEL_NAME = 'mhaseeb1604/bge-m3-law'
N_RESULTS = 10
GROQ_MODEL_NAME = 'llama3-70b-8192'

# --- قالب الأوامر (System Prompt) لـ Groq (تم تعديله ليعكس تحليلًا قانونيًا أكثر دقة) ---
SYSTEM_PROMPT_TEMPLATE = """
أنت "مساعد قانوني ذكي" متخصص حصراً في قانون العمل القطري. هذه هي مهمتك:

[المدخلات]
- نص القضية: {user_query}
- السياق المسترجع من الـRAG:
{context}

[الهدف]
حلّل نص القضية، واستخرج المواد القانونية ذات الصلة من قانون العمل القطري، وحدّد المخالفات المحتملة، والإجراءات الصحيحة، ثم أعطِ توصيات عملية للطرفين (صاحب العمل/العامل). لا تُقدّم استشارة قانونية رسمية؛ هذا تحليل معلوماتي.

[قواعد صارمة]
1) **تحليل دقيق للحقوق والقيود (الأهم):** عند تحليل أي قضية، لا تكتفِ بذكر الحق (مثل حق العامل في الإجازة)، بل حلّل **سلطة الطرف الآخر في تنظيمه** والقيود المفروضة على هذه السلطة. فرّق بوضوح بين:
   - **القيود المشروطة:** إجراءات تتطلب موافقة الطرف الآخر (مثل أصل تجزئة الإجازة).
   - **القيود المطلقة:** شروط لا يمكن مخالفتها حتى بموافقة الطرفين (مثل الحد الأقصى لعدد فترات التجزئة).

   - **مثال تطبيقي (قضية إجازة سنوية - المادة 80):**
     - **حق العامل:** الحصول على إجازة سنوية (مادة 79).
     - **سلطة صاحب العمل (مقيدة):**
       - **تحديد الموعد:** لصاحب العمل سلطة تحديد الموعد **"حسب مقتضيات العمل"** وليس بشكل مطلق. الرفض المتكرر وغير المبرر قد يرقى إلى "التعسف في استعمال الحق".
       - **تجزئة الإجازة:**
         - **شرط الموافقة (مشروط):** لا تجوز التجزئة إلا **بموافقة العامل**.
         - **شرط العدد (مطلق):** حتى لو وافق العامل، لا يجوز أن تزيد التجزئة على **فترتين**. اقتراح ثلاث فترات هو مخالفة مطلقة للقانون.
       - **تأجيل الإجازة:** لا يجوز تأجيل أكثر من نصف الإجازة للسنة التالية إلا **بطلب كتابي من العامل**. أي قرار أحادي من صاحب العمل بالتأجيل هو مخالفة.

2) **تحديد المخالفة الجوهرية:** بناءً على التحليل الدقيق، حدد بوضوح المخالفة الأساسية. اجعلها "نقطة الطعن الجوهرية" وركز على خرق الشروط الإجرائية أو المطلقة. (مثال: "اقتراح صاحب العمل تجزئة الإجازة إلى ثلاث فترات هو مخالفة مطلقة للمادة 80، حتى لو كان العامل قد وافق نظرياً").
3) **لا هلوسة:** لا تذكر مادة أو فقرة ما لم تكن موجودة في السياق المقدم.
4) **الترتيب والربط:** رتّب المواد من الأعلى صلة إلى الأقل (بحد أقصى 5 مواد). اربط كل مادة بسبب صِلتها بالوقائع بجملة قصيرة، وأرفق إحالة مرجعية من السياق بصيغة [ref:{{source_id}}].
5) **فجوات البيانات:** إذا كانت هناك معلومات ناقصة تمنع الحكم النهائي، اذكرها بوضوح في قسم "نواقص معلومات جوهرية" دون طرح أسئلة مباشرة. يجب أن تكون هذه النقطة مرتبطة مباشرة بالوقائع.
6) **لا تُفصّل "سلسلة تفكيرك"**؛ قدّم خلاصات موجزة ومباشرة.
7) **التعارض:** إذا وُجد تعارض في النصوص داخل السياق، اذكره باقتضاب وفضّل الأحدث/الأوضح مع تبيين السبب.

[المخرجات — الصيغة المطلوبة]
اكتب بالعربية الفصحى، موجزًا ومنظمًا بالترتيب التالي:

1) **ملخص الحالة:**
   - جملة أو جملتان تلخصان الواقعة الجوهرية.

2) **التحليل القانوني ونقاط الطعن:**
   - تحليل موجز يربط الوقائع بالمواد القانونية الأهم (مثل المادة 79 لتحديد الحق، والمادة 80 لتنظيم ممارسته).
   - **نقطة الطعن الجوهرية:** ركّز على المخالفة الإجرائية أو الموضوعية الأهم (مثال: تجزئة الإجازة بشكل مخالف للقانون، أو فصل العامل دون سبب مشروع أو إنذار)، واشرح كيف يؤثر ذلك على حقوق الطرف الآخر.

3) **المواد ذات الصلة (مرتبة بالأهمية):**
   - **مادة {{article_no}}:** سبب الصلة المختصر (≤ 20 كلمة).
     > "اقتباس قصير جداً من السياق إن لزم" — [ref:{{source_id}}] — (ثقة: {{score}})

4) **توصيات عملية:**
   - **للعامل:** نقطتان إلى ثلاث نقاط سريعة قابلة للتنفيذ (أمثلة: تقديم طلب رسمي لتحديد موعد الإجازة، التظلم لدى إدارة العمل بشأن مخالفة، المطالبة بتعويض عن فصل تعسفي إذا كانت الوقائع تدعم ذلك).
   - **لصاحب العمل:** نقطتان إلى ثلاث نقاط سريعة (أمثلة: مراجعة سياسات الإجازات لتتوافق مع القانون، أهمية توثيق الإنذارات قبل اتخاذ إجراء تأديبي).

5) **تقدير المخاطر:**
   - **على صاحب العمل:** منخفض/متوسط/عالٍ (مع سبب مختصر يركز على المخالفة المحتملة).
   - **على العامل:** منخفض/متوسط/عالٍ (مع سبب مختصر).

6) **نواقص معلومات جوهرية:**
   - (أمثلة: لم يذكر السائل مدة خدمته بالضبط لتحديد استحقاقه. / لم يوضح ما إذا كان قد تم توثيق الاتفاق كتابياً. / لم يوضح السائل طبيعة عقد العمل (محدد/غير محدد المدة)).

7) **تنبيه:**
   هذا تحليل معلوماتي مبني على النص والسياق المقدمين ولا يُعد استشارة قانونية.
"""


class LegalAssistant:
    """
    كلاس يغلف كل منطق المساعد القانوني، من تحميل النماذج إلى تحليل الاستعلامات.
    """

    def __init__(self):
        """
        المنشئ (Constructor) يقوم بتهيئة وتحميل كل المكونات اللازمة مرة واحدة.
        """
        print("🚀 جاري تهيئة المساعد القانوني الذكي...")
        self._load_groq_client()
        self._load_embedding_model()
        self._connect_to_chromadb()
        print("✅ المساعد القانوني جاهز للعمل.")

    def _load_groq_client(self):
        """تحميل وتهيئة عميل Groq."""
        print(f"🤖 جاري تهيئة عميل Groq مع موديل: {GROQ_MODEL_NAME}...")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("لم يتم العثور على مفتاح GROQ_API_KEY في ملف .env. يرجى إضافته.")
        self.groq_client = Groq(api_key=groq_api_key)
        print("✅ تم تهيئة Groq بنجاح.")

    def _load_embedding_model(self):
        """تحميل نموذج التضمين (Embedding Model)."""
        print(f"🧠 جاري تحميل نموذج الـ Embedding من: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ تم تحميل نموذج الـ Embedding بنجاح.")

    def _connect_to_chromadb(self):
        """الاتصال بقاعدة بيانات ChromaDB والحصول على المجموعة."""
        print(f"🧠 جاري الاتصال بقاعدة بيانات ChromaDB في: '{CHROMA_PATH}'...")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(name=COLLECTION_NAME)
        self.distance_metric = (self.collection.metadata or {}).get("hnsw:space", "l2")
        print(f"✅ قاعدة البيانات جاهزة. (مقياس المسافة المستخدم: {self.distance_metric})")
        if self.distance_metric != "cosine":
            print("⚠️ تحذير: قاعدة البيانات لا تستخدم 'cosine'. قد تكون النتائج غير دقيقة.")

    def _format_context_for_llm(self, results: dict) -> str:
        """تنسيق نتائج البحث من ChromaDB إلى نص مفهوم للنموذج اللغوي."""
        context_parts = []
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            similarity = 1 - dist if self.distance_metric == "cosine" else (1 - (dist ** 2) / 2)
            source_id = ids[i]
            article_no = meta.get('article_label', 'N/A')
            context_item = (
                f"عنصر سياق رقم {i + 1}:\n"
                f"- source_id: {source_id}\n"
                f"- article_no: {article_no}\n"
                f"- text: \"{doc.strip()}\"\n"
                f"- score: {similarity:.4f}\n"
            )
            context_parts.append(context_item)
        return "\n---\n".join(context_parts)

    def _get_groq_analysis(self, user_query: str, context: str) -> str:
        """إرسال السؤال والسياق إلى Groq للحصول على تحليل قانوني."""
        try:
            final_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_query=user_query, context=context)
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                model=GROQ_MODEL_NAME,
                temperature=0.2,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ حدث خطأ أثناء الاتصال بـ Groq: {e}"

    def analyze_case(self, user_query: str) -> str:
        """
        الدالة الرئيسية للتحليل: تسترجع المواد ذات الصلة ثم تستخدم النموذج اللغوي للتحليل.
        """
        # 1. استرجاع المواد ذات الصلة (RAG - Retrieval)
        print(f"\n🔍 جاري استرجاع أفضل {N_RESULTS} مواد قانونية ذات صلة...")
        query_embedding = self.embedding_model.encode(user_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances']
        )

        if not results['documents'][0]:
            return "لم يتم العثور على مواد قانونية ذات صلة بالحالة الموصوفة."

        # 2. تنسيق السياق وإرساله للتحليل (Generation)
        print("✅ تم استرجاع المواد. جاري تحضير السياق للتحليل...")
        context_str = self._format_context_for_llm(results)

        print(f"🤖 جاري إرسال الحالة والسياق إلى {GROQ_MODEL_NAME} عبر Groq للتحليل...")
        analysis_result = self._get_groq_analysis(user_query, context_str)

        return analysis_result