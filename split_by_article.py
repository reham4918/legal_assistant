# C:/Users/LENOVO/PycharmProjects/legal_assistant/split_by_article_v2.py

# -*- coding: utf-8 -*-
"""
الاستخدام:
  1) تأكد من تشغيل أوامر التثبيت النظيف في الطرفية (Terminal).
  2) ضع ملف القانون المطلوب في مجلد: data/
  3) شغّل: python split_by_article_v2.py
ينتج: data/qatar_labor_chunks.jsonl — سطر لكل مادة مع بيانات وصفية (metadata) مبسطة ومخصصة.
"""

import os
import re
import json
from typing import List, Dict, Any

# --- التحقق من إصدار LlamaIndex ---
try:
    from llama_index.core import __version__ as llama_index_version
    from packaging.version import Version

    if Version(llama_index_version) < Version("0.10.0"):
        raise ImportError(
            "هذا الكود يتطلب إصدار 0.10.0 أو أحدث من LlamaIndex. "
            "يرجى التحديث باستخدام الأمر: pip install -U llama-index"
        )
except (ImportError, ModuleNotFoundError):
    print("⚠️ لم يتم العثور على مكتبة `llama-index` أو `packaging`. يرجى تثبيتها: pip install -U llama-index packaging")
    exit(1)
# ------------------------------------


# --- مكونات LlamaIndex الأساسية ---
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import BaseExtractor

# -----------------------------------------------------------------

# ----------- إعدادات ثابتة (تم التعديل للتركيز على ملف واحد) -----------
PDF_FILE_PATH = "data/Law_2004_14(2).pdf"
OUT_PATH = "data/qatar_labor_chunks.jsonl"

LAW_ID = "قانون العمل القطري رقم (14) لسنة 2004"
# ------------------------------------

# --- أدوات مساعدة ---
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
DIGIT_MAP = {ord(a): b for a, b in zip(ARABIC_INDIC, WESTERN)}


def normalize_digits(text: str) -> str:
    """توحيد الأرقام الهندية إلى غربية."""
    return text.translate(DIGIT_MAP)


# --- التعابير النمطية (Regex) ---
ART_PAT = re.compile(
    r"^\s*المادة\s*[\(（]?\s*([0-9٠-٩]+)\s*[\)）]?"
    r"(?:\s*[-–—/]*\s*(مكرر[0-9٠-٩]*|إصدار))?\b",
    re.MULTILINE
)
CHAPTER_PAT = re.compile(r"^\s*الفصل.*?(?=\n\s*المادة|\Z)", re.MULTILINE | re.DOTALL)


# --- مستخرجات بيانات وصفية مخصصة (Custom Metadata Extractors) ---

class ChapterTitleExtractor(BaseExtractor):
    """
    مستخرج مخصص لتحديد عنوان الفصل الذي تنتمي إليه كل مادة.
    """
    def extract(self, nodes: List[BaseNode]) -> List[Dict]:
        metadata_list = []
        for node in nodes:
            text_before_node = node.metadata.get("text_before", "")
            matches = list(CHAPTER_PAT.finditer(text_before_node))
            raw_chapter = matches[-1].group(0).strip() if matches else ""

            if raw_chapter:
                chapter = re.sub(r'\s+', ' ', raw_chapter).strip()
            else:
                chapter = ""

            metadata_list.append({"chapter_title": chapter})
        return metadata_list

    async def aextract(self, nodes: List[BaseNode]) -> List[Dict]:
        return self.extract(nodes)


class ArticleInfoExtractor(BaseExtractor):
    """
    مستخرج مخصص لتحليل بداية كل مادة للحصول على معلوماتها.
    """
    def extract(self, nodes: List[BaseNode]) -> List[Dict]:
        metadata_list = []
        for node in nodes:
            match = ART_PAT.search(node.get_content())

            if match:
                raw_num = (match.group(1) or "").strip()
                suffix = (match.group(2) or "").strip()
                label = f"المادة {raw_num}" + (f" {suffix}" if suffix else "")

                current_chapter = node.metadata.get("chapter_title", "")
                if "إصدار" in label:
                    final_chapter = "مواد الإصدار"
                elif not current_chapter:
                    final_chapter = "بدون فصل"
                else:
                    final_chapter = current_chapter

                metadata_list.append({
                    "article_label": label,
                    "chapter_title": final_chapter
                })
            else:
                metadata_list.append({
                    "article_label": "N/A",
                    "chapter_title": "N/A"
                })
        return metadata_list

    async def aextract(self, nodes: List[BaseNode]) -> List[Dict]:
        return self.extract(nodes)


# --- دالة التصدير (تم تعديلها لإنشاء بيانات وصفية مبسطة) ---

def export_nodes_to_jsonl(nodes: List[BaseNode], out_path: str) -> Dict[str, Any]:
    """
    تقوم بكتابة القائمة النهائية من العقد (Nodes) في ملف JSONL.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, node in enumerate(nodes, start=1):
            full_meta = node.metadata
            label = full_meta.get("article_label", f"المادة_{i}")
            safe_id_part = re.sub(r'[\s\-/()（）]+', '_', label.replace("المادة", "art")).strip('_')
            chunk_id = f"QALaw2004-14_{safe_id_part}"

            clean_text = node.get_content(metadata_mode="none").strip()
            clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)

            # ✨ --- إنشاء قاموس بيانات وصفية مبسط --- ✨
            # هنا نقوم باختيار الحقول المطلوبة فقط
            minimal_metadata = {
                "law_id": full_meta.get("law_id", "N/A"),
                "chapter_title": full_meta.get("chapter_title", "N/A"),
                "article_label": full_meta.get("article_label", "N/A"),
            }

            rec = {
                "chunk_id": chunk_id,
                "text": clean_text,
                "metadata": minimal_metadata  # استخدام القاموس المبسط
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "articles": len(nodes),
        "chunks": len(nodes),
        "out_path": out_path,
    }


# -------------------- main --------------------

def main():
    # 1. تحميل المستند المحدد وتوحيد الأرقام
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ خطأ: لم يتم العثور على الملف المحدد في المسار: {PDF_FILE_PATH}")
        print("يرجى التأكد من وجود الملف وإعادة المحاولة.")
        return

    print(f"📖 جاري قراءة ملف PDF المحدد: {PDF_FILE_PATH}...")
    reader = SimpleDirectoryReader(input_files=[PDF_FILE_PATH])
    raw_documents = reader.load_data(show_progress=True)

    if not raw_documents:
        print(f"❌ فشل تحميل الملف: '{PDF_FILE_PATH}'.")
        return

    full_text = "\n".join([doc.text for doc in raw_documents])

    print("🔢 جاري تطبيع الأرقام في النص الكامل...")
    full_text = normalize_digits(full_text)

    # 2. التقسيم اليدوي الدقيق باستخدام Regex
    print(f"📄 جاري تقسيم النص إلى مواد...")
    matches = list(ART_PAT.finditer(full_text))

    nodes = []
    for idx, match in enumerate(matches):
        start_idx = match.start()
        end_idx = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        article_text = full_text[start_idx:end_idx]
        node = TextNode(text=article_text.strip())
        # إضافة البيانات المؤقتة اللازمة للمستخرجات فقط
        node.metadata["text_before"] = full_text[:start_idx]
        node.metadata["text_after"] = full_text[end_idx:]
        nodes.append(node)

    if not nodes:
        print("⚠️ لم يتم العثور على أي مادة. راجع ART_PAT أو تأكد من جودة استخراج النص.")
        return

    print(f"✅ تم تقسيم النص إلى {len(nodes)} مادة.")

    # 3. تشغيل مستخرجات البيانات الوصفية
    print("🔍 جاري استخراج البيانات الوصفية لكل مادة...")
    chapter_extractor = ChapterTitleExtractor()
    article_extractor = ArticleInfoExtractor()

    chapter_metadata_list = chapter_extractor.extract(nodes)
    for i, node in enumerate(nodes):
        node.metadata.update(chapter_metadata_list[i])

    article_metadata_list = article_extractor.extract(nodes)
    for i, node in enumerate(nodes):
        node.metadata.update(article_metadata_list[i])

    # 4. إضافة البيانات الوصفية الثابتة (المطلوبة فقط)
    print("🧹 جاري إضافة البيانات الوصفية النهائية...")
    for node in nodes:
        node.metadata.update({
            "law_id": LAW_ID,
        })

    # 5. تصدير النتائج النهائية
    print(f"💾 تصدير {len(nodes)} مادة إلى JSONL: {OUT_PATH}")
    stats = export_nodes_to_jsonl(nodes, OUT_PATH)

    # ملخص
    print("\n--- ✅ اكتملت المعالجة بنجاح ---")
    print(f"  عدد المواد: {stats['articles']}")
    print(f"  إجمالي الـ Chunks: {stats['chunks']}")
    print(f"  ملف الخرج: {stats['out_path']}")
    if nodes:
        print("\n— مثال Metadata لأول مادة (تم تبسيطها) —")
        # طباعة مثال من الملف الناتج مباشرة للتأكيد
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(json.dumps(json.loads(first_line)['metadata'], ensure_ascii=False, indent=2))
    print("---------------------------\n")


if __name__ == "__main__":
    main()