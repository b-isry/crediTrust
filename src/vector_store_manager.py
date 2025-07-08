# src/vector_store_manager.py

import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import torch
import argparse

# --- Configuration ---
DATA_PATH = './data/filtered_complaints.csv'
VECTOR_STORE_PATH = './vector_store'
COLLECTION_NAME = 'complaints'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32 # Batch size for the embedding model, adjust based on your RAM/VRAM

def main(sample_size=None):
    """
    Main function to build the vector store.
    It loads data, chunks text, generates embeddings, and indexes them in ChromaDB.
    """
    print("--- Starting Vector Store Creation ---")

    # 1. Load the cleaned dataset
    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['cleaned_narrative'], inplace=True)
        if sample_size:
            print(f"Using a sample of {sample_size} complaints.")
            df = df.sample(n=sample_size, random_state=42)
        print(f"Loaded {len(df)} complaints.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 2. Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    print(f"Initialized text splitter with chunk_size={CHUNK_SIZE} and chunk_overlap={CHUNK_OVERLAP}")

    # 3. Chunk all documents
    print("Chunking documents...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking complaints"):
        complaint_id = str(row['complaint_id'])
        product = row['product']
        narrative = row['cleaned_narrative']
        
        chunks = text_splitter.split_text(narrative)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                'complaint_id': complaint_id,
                'product': product,
                'chunk_index': i
            })
            all_ids.append(f"{complaint_id}_{i}")
    
    print(f"Created {len(all_chunks)} chunks from {len(df)} complaints.")

    # 4. Generate Embeddings using SentenceTransformer (with GPU if available)
    print("\n--- Generating Embeddings ---")
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the model directly
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    print("Generating embeddings for all chunks. This may take a while...")
    # The `encode` method is highly optimized and works in batches.
    embeddings = embedding_model.encode(
        all_chunks, 
        show_progress_bar=True, 
        batch_size=BATCH_SIZE
    )
    print(f"Generated {len(embeddings)} embeddings.")

    # 5. Initialize ChromaDB and Add Data
    print("\n--- Indexing in ChromaDB ---")
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # In Chroma, we don't need to specify the embedding function if we provide the embeddings directly
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add to ChromaDB in batches to avoid overwhelming the system
    db_batch_size = 5000
    for i in tqdm(range(0, len(all_chunks), db_batch_size), desc="Indexing Batches into ChromaDB"):
        collection.add(
            ids=all_ids[i:i+db_batch_size],
            embeddings=embeddings[i:i+db_batch_size].tolist(), # Pass pre-computed embeddings
            metadatas=all_metadatas[i:i+db_batch_size],
            documents=all_chunks[i:i+db_batch_size] # Still good to store the original text
        )

    print("\n--- Vector Store Creation Complete ---")
    print(f"Total documents indexed: {collection.count()}")
    print(f"Vector store persisted at: {VECTOR_STORE_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store for complaints.")
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=None, 
        help='Number of complaints to sample for a quick build. If not provided, the full dataset is used.'
    )
    args = parser.parse_args()
    
    main(sample_size=args.sample_size)