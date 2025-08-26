# C:/Users/LENOVO/PycharmProjects/legal_assistant/assistant_core.py

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
# ⚠️ تنبيه: هذا النموذج غير موجود على Groq وسيسبب خطأ.
# بناءً على طلبك، تم إبقاؤه كما هو.
# للحصول على نتائج، يجب تغييره إلى موديل صالح مثل 'llama3-70b-8192'.
GROQ_MODEL_NAME = 'openai/gpt-oss-20b'

# --- قالب الأوامر (System Prompt) لـ Groq (تم تحسينه بفرض سلسلة تفكير ومراجعة ذاتية) ---
SYSTEM_PROMPT_TEMPLATE = """
أنت "مساعد قانوني دقيق ومُحلل" متخصص حصراً في قانون العمل القطري. مهمتك هي تقديم تحليل قانوني مفصل ومتسق وخالٍ من التناقضات.

**مهمتك الأساسية:**
مهمتك هي تحديد ما إذا كان الاستعلام المقدم (`user_query`) **مكتفياً بذاته** لإجراء تحليل قانوني.

- **الاستعلام المكتفي بذاته:** هو الذي يحتوي على جميع الوقائع اللازمة لتحليل النقطة القانونية التي يثيرها.
  - في هذه الحالة، **يجب عليك تقديم تحليل قانوني كامل** فورًا، باستخدام السياق المسترجع (`context`) واتباع صيغة المخرجات والقواعد أدناه بدقة.

- **الاستعلام الناقص:** هو الذي يتعلق بموضوع معقد ويفتقر إلى تفاصيل جوهرية لا يمكن التحليل بدونها.
  - إذا كان الاستعلام ناقصًا، **يجب عليك تجاهل التحليل** والرد فقط بسؤال توضيحي لجمع المعلومات الناقصة.

**لا تشرح أبدًا خطوات تفكيرك. قدم مباشرة إما السؤال التوضيحي أو التحليل الكامل.**

---
**[قواعد التحليل الكامل الصارمة]**
(تُطبق فقط إذا كان الاستعلام مكتفياً بذاته)

1) **التحليل المنهجي (لتجنب التناقض):** عند تحليل نقطة قانونية، اتبع هذه الخطوات المنطقية الإلزامية بالترتيب:
    - **أ. تحديد القاعدة العامة:** ابدأ بذكر القاعدة العامة التي تنطبق على الموضوع (مثلاً، حق العامل في مكافأة نهاية الخدمة بموجب المادة 54).
    - **ب. تحديد القاعدة الخاصة (الاستثناء):** بعد ذلك، ابحث في السياق عن أي استثناءات أو شروط خاصة تنطبق على وقائع القضية (مثلاً، حالات الحرمان من المكافأة في المادة 61).
    - **ج. تطبيق الاستثناء بشكل حاسم:** وضح بعبارات صريحة لا لبس فيها أن القاعدة الخاصة (الاستثناء) هي التي تسري في هذه الحالة، وأنها **تقيّد وتلغي** تطبيق القاعدة العامة. يجب أن تكون هذه النقطة هي محور التحليل. مثال: "على الرغم من أن المادة 54 تمنح الحق في المكافأة بشكل عام، إلا أن المادة 61 تنص على استثناء واضح ينطبق هنا، وهو حرمان العامل من المكافأة في حالة الغياب المحدد. وبناءً عليه، فإن الاستثناء هو الذي يُطبق."
    - **د. الخلاصة النهائية:** بناءً على تطبيق الاستثناء، قدم خلاصة قانونية واحدة ومتسقة.

2) **المراجعة الذاتية الإلزامية:** بعد كتابة التحليل الأولي، قم بمراجعة إجابتك بالكامل. اسأل نفسك: "هل توصياتي وتقديري للمخاطر يتناقضان مع خلاصتي النهائية؟". إذا كانت الخلاصة هي أن العامل لا يستحق المكافأة، فيجب ألا تحتوي التوصيات على أي إشارة لمطالبة العامل بها. **يجب أن تكون جميع أجزاء الإجابة متسقة تمامًا مع الخلاصة النهائية.**

3) **الشفافية:** عند ذكر المواد، اذكر النص الكامل للمادة المسترجعة مع نسبة التشابه.

---
**[صيغة المخرجات للتحليل الكامل]**

1) **ملخص الحالة:**
   - جملة أو جملتان تلخصان الواقعة الجوهرية بدقة.

2) **التحليل القانوني ونقاط الطعن:**
   - **تحليل دقيق ومفصل:** (يجب أن يتبع القواعد الصارمة أعلاه، خاصة التحليل المنهجي والمراجعة الذاتية).

3) **المواد ذات الصلة (مرتبة بالأهمية):**
   - **لكل مادة مسترجعة من السياق، يجب عليك عرضها بالصيغة التالية:**
   - **مادة [رقم المادة] (نسبة التشابه: [قيمة score من السياق])**
   - > "[النص الكامل للمادة من حقل text في السياق]"

4) **توصيات عملية:**
   - **للعامل:** نقطتان إلى ثلاث نقاط **محددة وقابلة للتنفيذ** بناءً على التحليل الصحيح والمتسق.
   - **لصاحب العمل:** نقطتان إلى ثلاث نقاط **محددة وقابلة للتنفيذ** لتصحيح الوضع أو تجنب المشاكل مستقبلاً.

5) **تقدير المخاطر:**
   - **على صاحب العمل:** منخفض/متوسط/عالٍ (مع سبب مختصر ومباشر مرتبط بالخلاصة النهائية للتحليل).
   - **على العامل:** منخفض/متوسط/عالٍ (مع سبب مختصر مرتبط بقوة موقفه القانوني النهائية).

6) **نواقص معلومات جوهرية:**
   - (اذكر هنا النواقص الحقيقية والجوهرية فقط. إذا لم توجد، اذكر "لا توجد نواقص جوهرية").

7) **تنبيه:**
   هذا تحليل معلوماتي مبني على النص والسياق المقدمين ولا يُعد استشارة قانونية.

---
**[المدخلات الحالية]**
- نص القضية: {user_query}
- السياق المسترجع من الـRAG:
{context}

---
**[أمر حاسم ونهائي]**
**تذكر دائمًا: يجب أن تكون إجابتك النهائية بالكامل باللغة العربية الفصحى، بغض النظر عن أي شيء آخر. لا تستخدم أي كلمة إنجليزية في المخرجات النهائية.**
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
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
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
                temperature=0.1,
                # ✨ --- تم زيادة الحد الأقصى للتوكنز هنا --- ✨
                max_tokens=4096,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ حدث خطأ أثناء الاتصال بـ Groq: {e}"

    def analyze_case(self, user_query: str) -> str:
        """
        الدالة الرئيسية للتحليل: تسترجع المواد ذات الصلة ثم تستخدم النموذج اللغوي للتحليل.
        """
        print(f"\n🔍 جاري استرجاع أفضل {N_RESULTS} مواد قانونية ذات صلة...")
        query_embedding = self.embedding_model.encode(user_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances']
        )

        context_str = self._format_context_for_llm(results) if results['documents'][0] else "لم يتم العثور على سياق."

        print(f"🤖 جاري إرسال الحالة والسياق إلى {GROQ_MODEL_NAME} لاتخاذ القرار...")
        analysis_result = self._get_groq_analysis(user_query, context_str)

        return analysis_result