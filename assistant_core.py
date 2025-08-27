# -*- coding: utf-8 -*-
"""
assistant_core.py

This file contains the core logic for the legal assistant.
This logic has been encapsulated into a single class (`LegalAssistant`) to avoid
code duplication and to facilitate maintenance and development for both the
command-line interface (CLI) and the web interface (Flask).
"""

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Load environment variables from the .env file ---
load_dotenv()

# --- Core Configuration ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "qatar_labor_law"
EMBEDDING_MODEL_NAME = 'mhaseeb1604/bge-m3-law'
N_RESULTS = 10
GROQ_MODEL_NAME = 'openai/gpt-oss-20b'

# --- System Prompt Template for Groq (Optimized with Chain-of-Thought and Self-Correction) ---
# This prompt is intentionally left in Arabic as it directly instructs the LLM
# on how to generate the final, user-facing output in Arabic.
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
    A class that encapsulates all the logic for the legal assistant,
    from loading models to analyzing queries.
    """

    def __init__(self):
        """
        The constructor initializes and loads all necessary components once.
        """
        print("🚀 Initializing the Legal Assistant...")
        self._load_groq_client()
        self._load_embedding_model()
        self._connect_to_chromadb()
        print("✅ Legal Assistant is ready.")

    def _load_groq_client(self):
        """Loads and initializes the Groq client."""
        print(f"🤖 Initializing Groq client with model: {GROQ_MODEL_NAME}...")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")
        self.groq_client = Groq(api_key=groq_api_key)
        print("✅ Groq client initialized successfully.")

    def _load_embedding_model(self):
        """Loads the sentence-transformer embedding model."""
        print(f"🧠 Loading embedding model from: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded successfully.")

    def _connect_to_chromadb(self):
        """Connects to the ChromaDB database and gets the collection."""
        print(f"🧠 Connecting to ChromaDB at: '{CHROMA_PATH}'...")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Specify cosine distance
        )
        self.distance_metric = (self.collection.metadata or {}).get("hnsw:space", "l2")
        print(f"✅ Database is ready. (Distance metric: {self.distance_metric})")
        if self.distance_metric != "cosine":
            print("⚠️ Warning: The collection is not using 'cosine' distance. Similarity scores might be inaccurate.")

    def _format_context_for_llm(self, results: dict) -> str:
        """
        Formats the search results from ChromaDB into a string for the LLM.
        The context labels are intentionally in Arabic to match the system prompt.
        """
        context_parts = []
        # Ensure we are accessing the first (and only) list of results
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No context found."

        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            # Convert distance to a similarity score.
            # For cosine distance, similarity = 1 - distance.
            # For L2 (Euclidean) distance, a common conversion is 1 - (dist^2 / 2),
            # but cosine is preferred for this type of task.
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
        """Sends the query and context to Groq to get a legal analysis."""
        try:
            final_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_query=user_query, context=context)
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                model=GROQ_MODEL_NAME,
                temperature=0.1,
                max_tokens=4096,  # Increased token limit for detailed analysis
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"❌ An error occurred while contacting Groq: {e}")
            return f"An error occurred while contacting the AI model: {e}"

    def analyze_case(self, user_query: str) -> str:
        """
        The main analysis function: retrieves relevant articles and then uses
        the language model to generate an analysis.
        """
        print(f"\n🔍 Retrieving top {N_RESULTS} relevant legal articles...")
        query_embedding = self.embedding_model.encode(user_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=['metadatas', 'documents', 'distances']
        )

        context_str = self._format_context_for_llm(results)

        print(f"🤖 Sending case and context to {GROQ_MODEL_NAME} for analysis...")
        analysis_result = self._get_groq_analysis(user_query, context_str)

        return analysis_result