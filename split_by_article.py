# -*- coding: utf-8 -*-
"""
الاستخدام:
  1) تأكد من تثبيت المكتبات: pip install -U pypdf llama-index
  2) ضع هذا الملف داخل مشروعك، والـ PDF في: data/Law_2004_14.pdf
  3) شغّل: python split_by_article.py
ينتج: data/qatar_labor_chunks.jsonl — سطر لكل مادة مع بيانات وصفية (metadata) جاهزة لـ RAG.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# --- قراءة PDF (pypdf أولًا، ثم PyPDF2 كبديل) ---
try:
    from pypdf import PdfReader

    PDF_IMPL = "pypdf"
except ImportError:
    from PyPDF2 import PdfReader

    PDF_IMPL = "PyPDF2"

# --- LlamaIndex Document ---
from llama_index.core import Document

# ----------- إعدادات ثابتة -----------
PDF_PATH = "data/Law_2004_14.pdf"
OUT_PATH = "data/qatar_labor_chunks.jsonl"

LAW_ID = "قانون العمل القطري رقم (14) لسنة 2004"
LAW_TITLE = "قانون العمل"
LAW_YEAR = 2004
JURISDICTION = "QA"
DOC_TYPE = "qatar_labor_law_article"
LANGUAGE = "ar"
# ------------------------------------

# --- تطبيع الأرقام الهندية إلى العربية ---
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
DIGIT_MAP = {ord(a): b for a, b in zip(ARABIC_INDIC, WESTERN)}


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_MAP)


# --- علامة لتتبّع أرقام الصفحات ---
PAGE_MARK = "<<<PAGE:{}>>>"

# --- Regex: بداية المادة + دعم (مكرر/إصدار) ---
ART_PAT = re.compile(
    r"^\s*المادة\s*[\(（]?\s*([0-9٠-٩]+)\s*[\)）]?"
    r"(?:\s*[-–—/]*\s*(مكرر[0-9٠-٩]*|إصدار))?\b",
    re.MULTILINE
)

# --- Regex: عناوين الفصول (النسخة النهائية) ---
# هذا التعبير يلتقط السطر الذي يبدأ بـ "الفصل" وكل ما يليه من نصوص
# حتى يصل إلى بداية المادة التالية، مما يضمن التقاط العنوان فقط.
CHAPTER_PAT = re.compile(r"^\s*الفصل.*?(?=\n\s*المادة|\Z)", re.MULTILINE | re.DOTALL)


@dataclass
class Article:
    number: str
    text: str
    chapter_title: str
    page_start: int
    page_end: int
    number_int: Optional[int] = None
    is_bis: bool = False
    article_label: str = ""


# ---------------- دوال مساعدة ----------------

def load_pdf_with_page_marks(pdf_path: str) -> str:
    """قراءة PDF صفحة بصفحة مع وسم بداية كل صفحة لالتقاط page_start/page_end بدقة."""
    reader = PdfReader(pdf_path)
    buf = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        buf.append(PAGE_MARK.format(i))
        buf.append(normalize_digits(page_text))
    return "\n".join(buf)


def pages_in_span(span_text: str) -> Tuple[int, int]:
    pages = [int(m.group(1)) for m in re.finditer(r"<<<PAGE:(\d+)>>>", span_text)]
    return (min(pages), max(pages)) if pages else (1, 1)


def last_chapter_before(text_upto_idx: str) -> str:
    matches = list(CHAPTER_PAT.finditer(text_upto_idx))
    return matches[-1].group(0).strip() if matches else ""


def clean_span(span: str) -> str:
    """تنظيف نص المادة: إزالة علامات الصفحات، فك تقطيع الكلمات، إزالة سطر 'الفصل...' إن التصق."""
    t = span.replace("\r", "")
    t = re.sub(r"<<<PAGE:\d+>>>", "", t)
    # إزالة أي سطر فصل قد يلتصق داخل الجزء
    t = CHAPTER_PAT.sub("", t)
    # فك التقطيع: شرطة + سطر جديد
    t = re.sub(r"-\s*\n", "", t)
    # دمج كسور الكلمات الشائعة (حروف قبل/بعد الشرطة)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # تنضيف مسافات زائدة
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def parse_article_marker(m) -> tuple[Optional[int], bool, str]:
    """من مطابق ART_PAT يُستخرج: (number_int, is_bis, article_label)."""
    raw_num = (m.group(1) or "").strip()
    suffix = (m.group(2) or "").strip()  # "مكرر..." أو "إصدار" أو ""
    raw_num = normalize_digits(raw_num)
    num_m = re.search(r"\d+", raw_num)
    number_int = int(num_m.group(0)) if num_m else None
    is_bis = suffix.startswith("مكرر")
    label = f"المادة {raw_num}" + (f" {suffix}" if suffix else "")
    return number_int, is_bis, label


# ---------------- التقسيم إلى مواد ----------------

def split_by_articles(full_text: str) -> List[Article]:
    matches = list(ART_PAT.finditer(full_text))
    articles: List[Article] = []

    for idx, m in enumerate(matches):
        number_int, is_bis, label = parse_article_marker(m)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        span = full_text[start:end]

        # --- التعديل النهائي هنا ---
        # 1. استخلاص عنوان الفصل الخام
        raw_chapter = last_chapter_before(full_text[:start])

        # 2. تنظيف عنوان الفصل من الشوائب
        if raw_chapter:
            # إزالة علامات الصفحات
            temp_chapter = re.sub(r"<<<PAGE:\d+>>>", "", raw_chapter)
            # استبدال الأسطر الجديدة والمسافات الزائدة بمسافة واحدة
            chapter = re.sub(r'\s+', ' ', temp_chapter).strip()
        else:
            chapter = ""
        # --- نهاية التعديل ---

        # 3. التعامل مع الحالات الخاصة (مواد الإصدار)
        if not chapter and "إصدار" in label:
            chapter = "مواد الإصدار"
        elif not chapter:
            chapter = "بدون فصل"

        clean_text = clean_span(span)
        p_start, p_end = pages_in_span(span)

        articles.append(Article(
            number=str(number_int) if number_int is not None else label,
            text=clean_text,
            chapter_title=chapter,
            page_start=p_start,
            page_end=p_end,
            number_int=number_int,
            is_bis=is_bis,
            article_label=label
        ))
    return articles


# -------------- تحويل إلى Documents --------------

def articles_to_documents(articles: List[Article], source_file: str) -> List[Document]:
    docs: List[Document] = []
    for a in articles:
        metadata = {
            "law_id": LAW_ID,
            "law_title": LAW_TITLE,
            "law_year": LAW_YEAR,
            "jurisdiction": JURISDICTION,
            "language": LANGUAGE,
            "chapter_title": a.chapter_title,
            "article_number": a.number,  # كسلسلة (للإظهار)
            "article_number_int": a.number_int,  # كقيمة عددية (للفلترة)
            "is_bis": a.is_bis,
            "article_label": a.article_label or f"المادة {a.number}",
            "page_start": a.page_start,
            "page_end": a.page_end,
            "source_pdf": os.path.basename(source_file),
            "doc_type": DOC_TYPE,
        }
        docs.append(Document(text=a.text, metadata=metadata))
    return docs


# -------------- التصدير إلى JSONL --------------

def _num_key(meta: Dict[str, Any]) -> int:
    n = meta.get("article_number_int", None)
    return int(n) if isinstance(n, int) else 10 ** 9


def export_nodes_to_jsonl(nodes: List[Document], out_path: str) -> Dict[str, Any]:
    """
    تقوم هذه الدالة بكتابة القائمة النهائية من المواد في ملف JSONL.
    كل سطر في الملف يمثل مادة واحدة بصيغة JSON.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # فرز مستقر: رقم المادة (إن وجد) ثم is_bis ثم الصفحة
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (
            _num_key(n.metadata),
            1 if n.metadata.get("is_bis") else 0,
            n.metadata.get("page_start", 0)
        )
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for i, n in enumerate(sorted_nodes, start=1):
            meta = n.metadata

            # استخدام article_label مباشرة لإنشاء ID واضح وموثوق
            label = meta.get("article_label", f"المادة_{i}")
            # تحويل "المادة 52 - مكرر" إلى "art_52_مكرر"
            safe_id_part = re.sub(r'[\s\-/]+', '_', label.replace("المادة", "art"))
            chunk_id = f"QALaw2004-14_{safe_id_part}"

            rec = {
                "chunk_id": chunk_id,
                "text": n.text,
                "metadata": {
                    **meta,
                    "chunk_index": 1,
                    "chunks_in_article": 1,
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "articles": len(nodes),
        "chunks": len(nodes),
        "out_path": out_path,
        "pdf_reader_impl": PDF_IMPL
    }


# -------------------- main --------------------

def main():
    pdf_path = PDF_PATH
    if not os.path.exists(pdf_path):
        alt = pdf_path if pdf_path.lower().endswith(".pdf") else pdf_path + ".pdf"
        if os.path.exists(alt):
            pdf_path = alt
        else:
            print(f"❌ لم يتم العثور على ملف PDF عند: {PDF_PATH}")
            return

    print(f"بدء معالجة ملف PDF: {pdf_path} ({PDF_IMPL}) ...")
    full_text = load_pdf_with_page_marks(pdf_path)

    print(f"تقسيم النص إلى مواد باستخدام التعبير: {ART_PAT.pattern}")
    articles = split_by_articles(full_text)
    if not articles:
        print("⚠️ لم يتم العثور على أي مادة. راجع ART_PAT أو تأكد من جودة استخراج النص.")
        return

    print(f"تم العثور على {len(articles)} مادة. تحويلها إلى Documents...")
    documents = articles_to_documents(articles, pdf_path)

    print(f"تصدير {len(documents)} مادة إلى JSONL: {OUT_PATH}")
    stats = export_nodes_to_jsonl(documents, OUT_PATH)

    # ملخص
    print("\n--- اكتملت المعالجة ---")
    print(f"  مكتبة قراءة PDF: {stats['pdf_reader_impl']}")
    print(f"  عدد المواد: {stats['articles']}")
    print(f"  إجمالي الـ Chunks: {stats['chunks']}")
    print(f"  ملف الخرج: {stats['out_path']}")
    if documents:
        print("— مثال Metadata لأول مادة —")
        print(json.dumps(documents[0].metadata, ensure_ascii=False, indent=2))
    print("---------------------------\n")


if __name__ == "__main__":
    main()