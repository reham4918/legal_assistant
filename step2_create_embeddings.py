# -*- coding: utf-8 -*-
"""
Usage:
  1) Ensure the 'data/qatar_labor_chunks.jsonl' file exists.
  2) Install the required libraries: pip install sentence-transformers torch tqdm
  3) Run this script: python step2_create_embeddings.py

This script will generate a new file named 'qatar_labor_embeddings.jsonl' in the 'data' directory.
This new file will contain the same data as the source file, but with an added "embedding"
field for each legal article chunk.
"""

# --- Core Libraries ---
import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---

# Path to the source data file containing the text chunks.
INPUT_PATH = "data/qatar_labor_chunks.jsonl"
# Path where the new file with embeddings will be saved.
OUTPUT_PATH = "data/qatar_labor_embeddings.jsonl"

# The name of the model used to generate embeddings.
# This model (bge-m3-law) is a fine-tuned version of BGE-M3,
# specifically adapted for legal texts, making it an excellent choice for this project.
MODEL_NAME = 'mhaseeb1604/bge-m3-law'


# --- Helper Functions ---

def load_chunks(file_path: str) -> list[dict]:
    """
    Reads a JSONL file containing the legal article chunks.
    """
    # Check if the input file exists before attempting to open it.
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}.")
        print("Please make sure the file exists in the correct path.")
        return []

    # Open the file and load each line as a separate JSON object.
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_chunks_with_embeddings(file_path: str, data: list[dict]):
    """
    Saves the list of chunks, now including their embeddings, to a new JSONL file.
    """
    # Ensure the output directory ('data/') exists.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write each dictionary item as a new line in the JSONL file.
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Use ensure_ascii=False to correctly handle Arabic characters.
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# --- Main Execution Logic ---

def main():
    """
    The main function that orchestrates the script's workflow:
    1. Load the legal article chunks from the source file.
    2. Initialize the sentence-transformer embedding model.
    3. Generate embeddings for the text of each chunk.
    4. Add the embeddings to the data and save it to a new file.
    """
    # Step 1: Load the data from the JSONL file.
    print(f"📖 Loading chunks from: {INPUT_PATH}...")
    chunks = load_chunks(INPUT_PATH)
    if not chunks:
        return
    print(f"✅ Successfully loaded {len(chunks)} chunks.")

    # Step 2: Initialize the specialized sentence-transformer model.
    print(f"🧠 Initializing the embedding model: '{MODEL_NAME}'...")
    print("(This may take some time during the first run to download the model)")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model initialized successfully.")

    # Step 3: Generate embeddings for all text chunks.
    # Extract all text content to be processed in a single batch for efficiency.
    texts_to_embed = [chunk['text'] for chunk in chunks]
    print(f"⏳ Generating embeddings for {len(texts_to_embed)} text chunks...")

    # The model converts the list of texts into a list of embedding vectors.
    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,  # Display a progress bar in the console.
        convert_to_tensor=False  # Output as a numpy array for easier handling.
    )
    print("✅ Embeddings generated successfully.")

    # Step 4: Add embeddings to the data and save the new file.
    print(f"💾 Adding embeddings and saving the new file to: {OUTPUT_PATH}...")
    for i, chunk in enumerate(tqdm(chunks, desc="Updating data")):
        # The numpy array must be converted to a standard list to be JSON serializable.
        chunk['embedding'] = embeddings[i].tolist()

    save_chunks_with_embeddings(OUTPUT_PATH, chunks)

    # --- Final Summary ---
    print("\n--- ✨ Process Completed Successfully ✨ ---")
    print(f"  - Created embeddings for {len(chunks)} chunks.")
    print(f"  - New file saved at: {OUTPUT_PATH}")
    print("-------------------------------------\n")
    print("🚀 You are now ready to run `step3_load_to_chroma.py` to load the data into the vector database.")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()