import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os

# --- Configuration ---
DATA_PATH = './data/filtered_complaints.csv'
VECTOR_STORE_PATH = './vector_store'
COLLECTION_NAME = 'complaints'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def main():
    """
    Main function to build the vector store.
    It loads data, chunks text, generates embeddings, and indexes them in ChromaDB.
    """
    print("--- Starting Vector Store Creation ---")

    # 1. Load the cleaned dataset
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure we don't have nulls in the narrative column
        df.dropna(subset=['cleaned_narrative'], inplace=True)
        print(f"Loaded {len(df)} complaints from {DATA_PATH}")
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

    # 3. Initialize ChromaDB Client and Embedding Function
    # This will create the directory if it doesn't exist
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Get or create the collection, specifying the embedding function
    print(f"Initializing ChromaDB collection: '{COLLECTION_NAME}'")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} # Using cosine distance for similarity
    )

    # 4. Process and Add Documents in Batches
    print("Starting chunking, embedding, and indexing process...")
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing complaints"):
        complaint_id = str(row['complaint_id'])
        product = row['product']
        narrative = row['cleaned_narrative']

        # Chunk the narrative
        chunks = text_splitter.split_text(narrative)

        for i, chunk in enumerate(chunks):
            # The document is the text chunk itself
            all_chunks.append(chunk)
            
            # Metadata includes source info
            all_metadatas.append({
                'complaint_id': complaint_id,
                'product': product,
                'chunk_index': i # Store the chunk index within the complaint
            })
            
            # Create a unique ID for each chunk
            all_ids.append(f"{complaint_id}_{i}")

    # Add to ChromaDB in a single, large batch for efficiency
    # ChromaDB's `add` method handles the embedding generation via the specified embedding_function
    print(f"\nAdding {len(all_chunks)} chunks to the vector store. This may take a while...")
    
    # Batching the addition to avoid potential memory issues with very large datasets
    batch_size = 5000 
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing Batches"):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )

    print("\n--- Vector Store Creation Complete ---")
    print(f"Total documents indexed: {collection.count()}")
    print(f"Vector store persisted at: {VECTOR_STORE_PATH}")

if __name__ == '__main__':
    main()