import os
import re
import json
from typing import List, Dict, Any

# --- LlamaIndex Version Check ---
# Ensures that the installed version of LlamaIndex is compatible with this script.
try:
    from llama_index.core import __version__ as llama_index_version
    from packaging.version import Version

    if Version(llama_index_version) < Version("0.10.0"):
        raise ImportError(
            "This script requires LlamaIndex version 0.10.0 or newer. "
            "Please upgrade using: pip install -U llama-index"
        )
except (ImportError, ModuleNotFoundError):
    print(
        "⚠️ `llama-index` or `packaging` library not found. Please install them: pip install -U llama-index packaging")
    exit(1)

# --- Core LlamaIndex Components ---
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import BaseExtractor

# --- Configuration Constants ---
PDF_FILE_PATH = "data/Law_2004_14.pdf"
OUT_PATH = "data/qatar_labor_chunks.jsonl"
LAW_ID = "قانون العمل القطري رقم (14) لسنة 2004"

# --- Helper Tools for Text Normalization ---
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
DIGIT_MAP = {ord(a): b for a, b in zip(ARABIC_INDIC, WESTERN)}


def normalize_digits(text: str) -> str:
    """Converts Arabic-Indic digits to Western digits for consistency."""
    return text.translate(DIGIT_MAP)


# --- Regular Expressions for Text Parsing ---
# Pattern to identify the start of a law article (e.g., "المادة 1", "المادة 2 مكرر").
ART_PAT = re.compile(
    r"^\s*المادة\s*[\(（]?\s*([0-9٠-٩]+)\s*[\)）]?"
    r"(?:\s*[-–—/]*\s*(مكرر[0-9٠-٩]*|إصدار))?\b",
    re.MULTILINE
)
# Pattern to identify chapter titles (e.g., "الفصل الأول").
CHAPTER_PAT = re.compile(r"^\s*الفصل.*?(?=\n\s*المادة|\Z)", re.MULTILINE | re.DOTALL)


# --- Custom Metadata Extractors ---

class ChapterTitleExtractor(BaseExtractor):
    """
    A custom extractor to find the chapter title for each law article.
    It looks for the last chapter title that appeared before the article text.
    """

    def extract(self, nodes: List[BaseNode]) -> List[Dict]:
        metadata_list = []
        for node in nodes:
            # Get the text that appeared before this node, stored in temporary metadata.
            text_before_node = node.metadata.get("text_before", "")
            matches = list(CHAPTER_PAT.finditer(text_before_node))
            # The relevant chapter is the last one found before the article.
            raw_chapter = matches[-1].group(0).strip() if matches else ""

            if raw_chapter:
                # Clean up whitespace for a consistent format.
                chapter = re.sub(r'\s+', ' ', raw_chapter).strip()
            else:
                chapter = ""

            metadata_list.append({"chapter_title": chapter})
        return metadata_list

    async def aextract(self, nodes: List[BaseNode]) -> List[Dict]:
        # Asynchronous version simply calls the synchronous one.
        return self.extract(nodes)


class ArticleInfoExtractor(BaseExtractor):
    """
    A custom extractor to parse the article's label (e.g., "المادة 61")
    and finalize the chapter title.
    """

    def extract(self, nodes: List[BaseNode]) -> List[Dict]:
        metadata_list = []
        for node in nodes:
            match = ART_PAT.search(node.get_content())

            if match:
                raw_num = (match.group(1) or "").strip()
                suffix = (match.group(2) or "").strip()
                label = f"المادة {raw_num}" + (f" {suffix}" if suffix else "")

                # Use the chapter title extracted by the previous extractor.
                current_chapter = node.metadata.get("chapter_title", "")
                # Handle special cases for chapter titles.
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
                # Fallback if no article pattern is matched.
                metadata_list.append({
                    "article_label": "N/A",
                    "chapter_title": "N/A"
                })
        return metadata_list

    async def aextract(self, nodes: List[BaseNode]) -> List[Dict]:
        # Asynchronous version simply calls the synchronous one.
        return self.extract(nodes)


# --- Export Function ---

def export_nodes_to_jsonl(nodes: List[BaseNode], out_path: str) -> Dict[str, Any]:
    """
    Writes the final list of nodes to a JSONL file.
    Each line in the file is a JSON object representing one article chunk.
    """
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, node in enumerate(nodes, start=1):
            full_meta = node.metadata
            label = full_meta.get("article_label", f"المادة_{i}")

            # Create a safe, unique chunk ID from the article label.
            safe_id_part = re.sub(r'[\s\-/()（）]+', '_', label.replace("المادة", "art")).strip('_')
            chunk_id = f"QALaw2004-14_{safe_id_part}"

            # Clean the node's text content for export.
            clean_text = node.get_content(metadata_mode="none").strip()
            clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)

            # Create a simplified metadata dictionary for the final output.
            # This selects only the necessary fields for the RAG system.
            minimal_metadata = {
                "law_id": full_meta.get("law_id", "N/A"),
                "chapter_title": full_meta.get("chapter_title", "N/A"),
                "article_label": full_meta.get("article_label", "N/A"),
            }

            # Construct the final record and write it as a JSON line.
            rec = {
                "chunk_id": chunk_id,
                "text": clean_text,
                "metadata": minimal_metadata
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Return statistics about the export process.
    return {
        "articles": len(nodes),
        "chunks": len(nodes),
        "out_path": out_path,
    }


# -------------------- Main Execution Block --------------------

def main():
    # Step 1: Load and Normalize the Document
    # Check if the source PDF file exists.
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ Error: The specified file was not found at: {PDF_FILE_PATH}")
        print("Please ensure the file exists and try again.")
        return

    # Use LlamaIndex's SimpleDirectoryReader to load the PDF content.
    print(f"📖 Reading the specified PDF file: {PDF_FILE_PATH}...")
    reader = SimpleDirectoryReader(input_files=[PDF_FILE_PATH])
    raw_documents = reader.load_data(show_progress=True)

    if not raw_documents:
        print(f"❌ Failed to load the file: '{PDF_FILE_PATH}'.")
        return

    # Combine the text from all pages into a single string.
    full_text = "\n".join([doc.text for doc in raw_documents])

    # Normalize all digits in the text to Western numerals for consistent matching.
    print("🔢 Normalizing digits in the full text...")
    full_text = normalize_digits(full_text)

    # Step 2: Manually Split the Text into Articles using Regex
    print(f"📄 Splitting the text into articles...")
    matches = list(ART_PAT.finditer(full_text))

    nodes = []
    for idx, match in enumerate(matches):
        start_idx = match.start()
        # Determine the end of the article by finding the start of the next one.
        end_idx = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        article_text = full_text[start_idx:end_idx]

        # Create a LlamaIndex TextNode for each article.
        node = TextNode(text=article_text.strip())

        # Add temporary metadata needed only by the custom extractors.
        node.metadata["text_before"] = full_text[:start_idx]
        node.metadata["text_after"] = full_text[end_idx:]
        nodes.append(node)

    if not nodes:
        print("⚠️ No articles were found. Check the ART_PAT regex or the text extraction quality.")
        return

    print(f"✅ Successfully split the text into {len(nodes)} articles.")

    # Step 3: Run Custom Metadata Extractors
    print("🔍 Extracting metadata for each article...")
    chapter_extractor = ChapterTitleExtractor()
    article_extractor = ArticleInfoExtractor()

    # Extract chapter titles and update node metadata.
    chapter_metadata_list = chapter_extractor.extract(nodes)
    for i, node in enumerate(nodes):
        node.metadata.update(chapter_metadata_list[i])

    # Extract article labels and finalize chapter titles.
    article_metadata_list = article_extractor.extract(nodes)
    for i, node in enumerate(nodes):
        node.metadata.update(article_metadata_list[i])

    # Step 4: Add Static Metadata to All Nodes
    print("🧹 Adding final metadata...")
    for node in nodes:
        node.metadata.update({
            "law_id": LAW_ID,
        })

    # Step 5: Export the Processed Nodes to a JSONL File
    print(f"💾 Exporting {len(nodes)} articles to JSONL: {OUT_PATH}")
    stats = export_nodes_to_jsonl(nodes, OUT_PATH)

    # --- Final Summary ---
    print("\n--- ✅ Processing Completed Successfully ---")
    print(f"  Number of Articles: {stats['articles']}")
    print(f"  Total Chunks: {stats['chunks']}")
    print(f"  Output File: {stats['out_path']}")
    if nodes:
        print("\n— Example Metadata for the first article (simplified) —")
        # Read and print the metadata of the first line from the output file to confirm.
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(json.dumps(json.loads(first_line)['metadata'], ensure_ascii=False, indent=2))
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()