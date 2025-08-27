# -*- coding: utf-8 -*-
"""
Usage:
  1) Ensure that step2_create_embeddings.py has been run first to generate
     the 'qatar_labor_embeddings.jsonl' file.
  2) Install the necessary libraries: pip install chromadb tqdm
  3) Run this script: python step3_load_to_chroma.py

This script will create a persistent vector database in a 'chroma_db' directory,
storing and indexing all the legal articles along with their embeddings.
"""

# --- Core Libraries ---
import json
import os
import chromadb
from tqdm import tqdm

# --- Key Configuration Settings ---

# Path to the data file containing the chunks and their embeddings.
INPUT_PATH = "data/qatar_labor_embeddings.jsonl"
# The directory where the vector database will be stored on disk.
CHROMA_PATH = "chroma_db"
# The name of the collection within the database where the data will be stored.
COLLECTION_NAME = "qatar_labor_law"


# --- Helper Functions ---

def load_embeddings_data(file_path: str) -> list[dict]:
    """
    Reads the JSONL file that contains the text chunks and their embeddings.
    """
    # First, check if the required input file exists.
    if not os.path.exists(file_path):
        print(f"❌ Error: The file {file_path} was not found.")
        print("Please make sure to run `step2_create_embeddings.py` first.")
        return []

    # Open the file and parse each line as a separate JSON object.
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# --- Main Execution Function ---

def main():
    """
    The main function that orchestrates the script's workflow:
    1. Load the data with its embeddings from the JSONL file.
    2. Initialize the ChromaDB client.
    3. Create or get the specified collection.
    4. Add the data to the collection.
    """
    # Step 1: Load the data from the source file.
    print(f"📖 Loading articles and embeddings from: {INPUT_PATH}...")
    chunks = load_embeddings_data(INPUT_PATH)
    # If loading fails or the file is empty, exit the script.
    if not chunks:
        return
    print(f"✅ Successfully loaded {len(chunks)} articles.")

    # Step 2: Initialize the ChromaDB client.
    # We use PersistentClient to save the database to the specified disk path.
    print(f"🧠 Initializing ChromaDB database at: '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Step 3: Create or get the collection.
    # `get_or_create_collection` is idempotent, meaning it won't cause an error
    # if the script is run multiple times.
    print(f"🔍 Creating or getting collection named: '{COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("✅ Collection retrieved successfully.")

    # Step 4: Add the data to the ChromaDB collection.
    # We need to prepare the data in the format ChromaDB expects:
    # separate lists for ids, documents, metadatas, and embeddings.
    print(f"➕ Adding {len(chunks)} articles to the database...")

    # Extract the data into separate lists.
    ids = [chunk['chunk_id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    embeddings = [chunk['embedding'] for chunk in chunks]

    # Add the data in a single batch.
    # For very large datasets, it's better to add data in smaller batches.
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # --- Final Summary ---
    # Print a summary of the operation to confirm success.
    count = collection.count()
    print("\n--- ✨ Process Completed Successfully ✨ ---")
    print(f"  - Added and indexed {count} articles in the database.")
    print(f"  - Collection Name: {COLLECTION_NAME}")
    print(f"  - Database saved in: {CHROMA_PATH}")
    print("-------------------------------------\n")
    print("🚀 You are now ready to build the RAG system and start asking questions!")


if __name__ == "__main__":
    main()