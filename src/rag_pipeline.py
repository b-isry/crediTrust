# src/rag_pipeline.py

import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
VECTOR_STORE_PATH = './vector_store'
COLLECTION_NAME = 'complaints'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# Note: Using a powerful model like Mistral-7B. This requires significant resources.
# For local testing on machines without a powerful GPU, consider a smaller model
# or a quantized version (e.g., from TheBloke on Hugging Face).
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

class RAGPipeline:
    def __init__(self):
        """
        Initializes the RAG pipeline by loading the necessary models and connecting to the vector store.
        """
        print("Initializing RAG Pipeline...")
        
        # Check for GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # 1. Load the retriever components
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        print("Connecting to vector store...")
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        self.collection = client.get_collection(name=COLLECTION_NAME)
        
        # 2. Load the generator components (LLM)
        print(f"Loading LLM: {LLM_MODEL_NAME}")
        # Use bfloat16 for memory efficiency if on a capable GPU
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        # Note: If you have memory issues, you may need to load in 8-bit or 4-bit mode.
        # This requires `bitsandbytes` and `accelerate` libraries.
        # model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch_dtype, load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch_dtype)
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1, # Use device_map="auto" if you have multiple GPUs
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.95
        )
        print("RAG Pipeline initialized successfully.")

    def retrieve_context(self, question: str, top_k: int = 5) -> list[str]:
        """
        Retrieves the top-k most relevant document chunks from the vector store.
        """
        print(f"Retrieving top {top_k} contexts for question: '{question}'")
        query_embedding = self.embedding_model.encode(question, convert_to_tensor=False).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = results['documents'][0]
        # Optional: Log or inspect results['distances'][0] to see relevance scores
        return retrieved_docs

    def generate_answer(self, question: str, context: list[str]) -> str:
        """
        Generates an answer using the LLM based on the provided question and context.
        """
        print("Generating answer with LLM...")
        
        # This prompt template is crucial for guiding the LLM
        prompt_template = """
        You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
        Use only the following retrieved complaint excerpts to formulate your answer.
        The answer should be a concise summary. Do not make up information.
        If the context doesn't contain enough information to answer the question, state that you cannot answer based on the provided context.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        
        # Format the context for the prompt
        formatted_context = "\n---\n".join(context)
        
        formatted_prompt = prompt_template.format(context=formatted_context, question=question)
        
        # The pipeline returns a list of dictionaries
        llm_response = self.llm_pipeline(formatted_prompt)
        
        # Extract the generated text
        generated_text = llm_response[0]['generated_text']
        
        # The response includes the prompt, so we extract only the part after "ANSWER:"
        answer = generated_text.split("ANSWER:")[1].strip()
        
        return answer

    def query(self, question: str) -> dict:
        """
        The main entry point for asking a question to the RAG system.
        Orchestrates retrieval and generation.
        """
        retrieved_context = self.retrieve_context(question)
        
        if not retrieved_context:
            return {
                "answer": "I could not find any relevant information in the complaint database to answer your question.",
                "sources": []
            }
        
        answer = self.generate_answer(question, retrieved_context)
        
        return {
            "answer": answer,
            "sources": retrieved_context
        }

if __name__ == '__main__':
    # This is for testing the module directly
    print("--- Running a test query ---")
    rag = RAGPipeline()
    
    test_question = "Why are people unhappy with the BNPL service?"
    result = rag.query(test_question)
    
    print("\n" + "="*50)
    print(f"QUESTION: {test_question}")
    print(f"\nGENERATED ANSWER:\n{result['answer']}")
    print("\n--- SOURCES USED ---")
    for i, source in enumerate(result['sources']):
        print(f"Source {i+1}:\n{source}\n")
    print("="*50)