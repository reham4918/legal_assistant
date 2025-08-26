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
GROQ_MODEL_NAME = 'llama3-70b-8192'

# --- قالب الأوامر (System Prompt) لـ Groq (تم تحسينه ليكون أكثر اهتمامًا بالتفاصيل) ---
SYSTEM_PROMPT_TEMPLATE = """
أنت "مساعد قانوني دقيق ومُحلل" متخصص حصراً في قانون العمل القطري. مهمتك هي تقديم تحليل قانوني مفصل ومبني على الأدلة.

**مهمتك الأساسية:**
مهمتك هي تحديد ما إذا كان الاستعلام المقدم (`user_query`) **مكتفياً بذاته** لإجراء تحليل قانوني.

- **الاستعلام المكتفي بذاته:** هو الذي يحتوي على جميع الوقائع اللازمة لتحليل النقطة القانونية التي يثيرها.
  - **مثال على استعلام مكتفٍ بذاته:** "عند انتهاء عقدي، رفضت الشركة تسليمي شهادة الخدمة". هذا الاستعلام كامل ولا يتطلب أي معلومات إضافية لتحليله في ضوء المادة 53.
  - في هذه الحالة، **يجب عليك تقديم تحليل قانوني كامل** فورًا، باستخدام السياق المسترجع (`context`) واتباع صيغة المخرجات المحددة أدناه بدقة.

- **الاستعلام الناقص:** هو الذي يتعلق بموضوع معقد ويفتقر إلى تفاصيل جوهرية لا يمكن التحليل بدونها.
  - **الحالات التي تتطلب تفاصيل إضافية بشكل شبه دائم:**
    - **قضايا الفصل:** تتطلب معرفة (سبب الفصل، مدة الخدمة، نوع العقد، وهل تم توجيه إنذار).
    - **قضايا مكافأة نهاية الخدمة:** تتطلب معرفة (مدة الخدمة الإجمالية، سبب انتهاء العقد، ونوع العقد).
    - **قضايا الإجازات:** تتطلب معرفة (مدة الخدمة لتحديد الاستحقاق).
  - إذا كان الاستعلام يندرج تحت هذه الفئة وهو ناقص، **يجب عليك تجاهل التحليل** والرد فقط بسؤال توضيحي لجمع المعلومات الناقصة. مثال: "أتفهم أن لديك استفسارًا بخصوص الفصل. للحصول على تحليل دقيق، يرجى توضيح بعض التفاصيل مثل: ما هو سبب الفصل المذكور من قبل صاحب العمل؟ وما هي مدة خدمتك؟"

**القاعدة الذهبية:**
لا تسأل عن تفاصيل لا تحتاجها. إذا كان بإمكانك الإجابة على السؤال القانوني المطروح بالمعلومات المتوفرة، فقم بذلك.

**لا تشرح أبدًا خطوات تفكيرك. قدم مباشرة إما السؤال التوضيحي أو التحليل الكامل.**

---
**[صيغة المخرجات للتحليل الكامل]**
(تُستخدم فقط إذا كان الاستعلام مكتفياً بذاته)

1) **ملخص الحالة:**
   - جملة أو جملتان تلخصان الواقعة الجوهرية بدقة.

2) **التحليل القانوني ونقاط الطعن:**
   - **تحليل دقيق ومفصل:** ابدأ بتفكيك وقائع القضية التي ذكرها المستخدم نقطة بنقطة. لكل واقعة، اربطها بالمادة القانونية المناسبة من السياق، موضحًا الحقوق والالتزامات المترتبة عليها. بعد ذلك، حدد **نقطة الطعن الجوهرية** بوضوح، وهي المخالفة الأساسية التي ارتكبها أحد الطرفين.

3) **المواد ذات الصلة (مرتبة بالأهمية):**
   - **مادة {{article_no}}:** سبب الصلة المختصر. > "اقتباس قصير ودقيق من المادة" — [ref:{{source_id}}] — (ثقة: {{score}})

4) **توصيات عملية:**
   - **للعامل:** نقطتان إلى ثلاث نقاط **محددة وقابلة للتنفيذ** بناءً على التحليل.
   - **لصاحب العمل:** نقطتان إلى ثلاث نقاط **محددة وقابلة للتنفيذ** لتصحيح الوضع أو تجنب المشاكل مستقبلاً.

5) **تقدير المخاطر:**
   - **على صاحب العمل:** منخفض/متوسط/عالٍ (مع سبب مختصر ومباشر مرتبط بنقطة الطعن الجوهرية).
   - **على العامل:** منخفض/متوسط/عالٍ (مع سبب مختصر مرتبط بقوة موقفه القانوني).

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
                max_tokens=2048,
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