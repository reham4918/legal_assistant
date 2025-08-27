# Legal Assistant - Qatar Labor Law RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing Qatar Labor Law with a web interface. This system processes legal documents, creates vector embeddings, and provides intelligent legal analysis using AI.

## 📁 Project Structure
```
legal_assistant/
├── app.py                          # Web interface
├── assistant_core.py               # RAG engine
├── step1_split_by_article.py       # PDF processing
├── step2_create_embeddings.py      # Embedding generation
├── step3_load_to_chroma.py         # Database setup
├── requirements.txt                # Python dependencies
├── data/
│   ├── Law_2004_14.pdf            # Source PDF
│   ├── qatar_labor_chunks.jsonl   # Processed articles
│   └── qatar_labor_embeddings.jsonl # Articles with embeddings
├── chroma_db/                      # Vector database
├── templates/
│   ├── index.html                 # Query form
│   └── results.html               # Results display
└── .env                           # API keys
```

## 🚀 Quick Start

**Prerequisites:**
- Python 3.8+
- PDF file of Qatar Labor Law (placed in `data/` directory)
- Groq API key (get from [Groq](https://console.groq.com/))

**Setup:**
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with: `GROQ_API_KEY=your_groq_api_key_here`
3. **Option A - Quick Start:** The processed data and ChromaDB database are already generated, so you can skip directly to:
   ```bash
   python app.py
   ```
4. **Option B - Full Pipeline:** If you want to regenerate the data or run the pipeline on your own data:
   ```bash
   python step1_split_by_article.py
   python step2_create_embeddings.py  
   python step3_load_to_chroma.py
   python app.py
   ```

## 📋 File Overview and Execution Order

### 1. `step1_split_by_article.py` - Document Processing
**Purpose:** Splits the Qatar Labor Law PDF into individual articles and extracts metadata.

**How to run:**
```bash
python step1_split_by_article.py
```

**What it does:**
- Reads PDF from `data/Law_2004_14.pdf`
- Uses regex patterns to identify and split articles (المادة 1, المادة 2, etc.)
- Extracts chapter information and article labels
- Normalizes Arabic-Indic digits to Western digits
- Outputs structured data to `data/qatar_labor_chunks.jsonl`

**Main Components:**
- `normalize_digits()`: Converts Arabic numerals to Western format
- `ChapterTitleExtractor`: Custom LlamaIndex extractor for chapter titles
- `ArticleInfoExtractor`: Extracts article labels and metadata
- `export_nodes_to_jsonl()`: Saves processed articles to JSONL format

**Configuration:**
- `PDF_FILE_PATH`: Path to the input PDF file
- `OUT_PATH`: Output path for processed chunks
- `ART_PAT`: Regex pattern for article detection

---

### 2. `step2_create_embeddings.py` - Embedding Generation
**Purpose:** Creates vector embeddings for each legal article using a specialized legal language model.

**How to run:**
```bash
python step2_create_embeddings.py
```

**What it does:**
- Loads article chunks from `data/qatar_labor_chunks.jsonl`
- Uses `mhaseeb1604/bge-m3-law` model (legal-specialized BGE-M3)
- Generates high-dimensional vector embeddings for semantic search
- Outputs enriched data to `data/qatar_labor_embeddings.jsonl`

**Main Components:**
- `load_chunks()`: Reads JSONL input file
- `save_chunks_with_embeddings()`: Saves data with embeddings
- `main()`: Orchestrates the embedding generation process

**Configuration:**
- `MODEL_NAME`: Embedding model (default: `mhaseeb1604/bge-m3-law`)
- `INPUT_PATH`: Source chunks file
- `OUTPUT_PATH`: Output file with embeddings

---

### 3. `step3_load_to_chroma.py` - Vector Database Setup
**Purpose:** Loads embedded articles into ChromaDB vector database for fast similarity search.

**How to run:**
```bash
python step3_load_to_chroma.py
```

**What it does:**
- Reads embeddings from `data/qatar_labor_embeddings.jsonl`
- Creates persistent ChromaDB database in `chroma_db/` directory
- Stores articles with metadata for retrieval
- Enables semantic search across legal articles

**Main Components:**
- `load_embeddings_data()`: Loads embeddings from JSONL
- `main()`: Sets up ChromaDB and adds data to collection

**Configuration:**
- `CHROMA_PATH`: Database storage directory
- `COLLECTION_NAME`: ChromaDB collection name
- `INPUT_PATH`: Embeddings input file

---

### 4. `assistant_core.py` - RAG Engine
**Purpose:** Core logic for the legal assistant, handling retrieval and AI analysis.

**Main Class: `LegalAssistant`**
- `__init__()`: Initializes all components (Groq client, embedding model, ChromaDB)
- `analyze_case()`: Main analysis function that processes user queries
- `_get_groq_analysis()`: Sends context to AI for legal analysis
- `_format_context_for_llm()`: Formats retrieved articles for the AI

**Configuration:**
- `EMBEDDING_MODEL_NAME`: Model for query embeddings
- `GROQ_MODEL_NAME`: AI model for analysis
- `TEMPERATURE`: Temperature for the LLM
- `MAX_TOKENS`: Maximum number of tokens for the LLM
- `N_RESULTS`: Number of articles to retrieve
- `SYSTEM_PROMPT_TEMPLATE`: Detailed prompt for legal analysis

---

### 5. `app.py` - Web Interface
**Purpose:** Flask web application providing user-friendly interface for legal queries.

**How to run:**
```bash
python app.py
```

**What it does:**
- Serves web interface at `http://localhost:5000`
- Handles user queries and displays formatted results
- Converts Markdown responses to HTML for better presentation
- Provides error handling for system failures

**Main Routes:**
- `/`: Home page with query form
- `/analyze`: Processes legal queries and returns analysis

**Main Components:**
- Flask app initialization and route handling
- Integration with `LegalAssistant` class
- Markdown to HTML conversion for formatted output


## 📝 Example Queries (in Arabic)

- "تم فصلي من العمل بسبب غيابي الغير مبرر لمدة 15 يوم بشكل متقطع ، لم يتم انذاري كتابيا ، لم يتم دفع مكافأة نهاية الخدمة ، عقدي محدود المدة ، مدة خدمتي 5 سنوات"
- "مهندس تم فصله فوريًا بدعوى إفشاء أسرار الشركة الفنية لعميل منافس. ربّ العمل يستند لنص الفصل دون إنذار، والعامل يقرّ أن بريدًا سُرّب خطأً لكنه لم يفشِ متعمدًا. النزاع يدور حول توافر سبب الفصل الجسيم وواجب كتمان الأسرار"
- "موظف أكمل سنة خدمة ويرغب في إجازته السنوية كاملة. صاحب العمل حاول تجزئتها إلى ثلاث فترات قصيرة متباعدة بدافع ضغط العمل، ورفض تحديد موعد معقول. الموظف يطلب الحد الأدنى المستحق (3 أسابيع إذا كانت خدمته أقل من 5 سنوات/4 أسابيع إن بلغت 5 فأكثر) ويحتج بأن التجزئة لا تكون لأكثر من فترتين وأن لصاحب العمل حق تحديد الموعد ضمن حدود المعقول."
- "انتهاء علاقة العمل، رفضت المنشأة تسليم العامل شهادة خبرة مفصلة بمدة خدمته ووظيفته وأجره، ورفضت أيضًا تذكرة العودة إلى بلده الأصلي كما اتُفق عند الاستقدام. العامل يطالب بالشهادة وردّ مستنداته الرسمية وتحمل نفقات إعادته."
