# src/rag_pipeline.py (API Version)

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
VECTOR_STORE_PATH = './vector_store'
COLLECTION_NAME = 'complaints'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline (API Version)...")
        
        # 1. Initialize Retriever
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        self.collection = client.get_collection(name=COLLECTION_NAME)

        # 2. Initialize LLM (via Groq API)
        # This is lightweight and just points to the API
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

        # 3. Define the RAG Chain using LangChain Expression Language (LCEL)
        prompt_template = """
        You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
        Use only the following retrieved complaint excerpts to formulate your answer.
        The answer should be a concise summary. Do not make up information.
        If the context doesn't contain enough information to answer, state that.

        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n---\n".join(docs)
        
        # This chain ties everything together
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG Pipeline initialized successfully.")

    def retriever(self, question: str) -> list[str]:
        """The retriever function for the LangChain chain."""
        query_embedding = self.embedding_model.encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
        return results['documents'][0]

    def query(self, question: str) -> dict:
        """Main query method that invokes the RAG chain."""
        answer = self.rag_chain.invoke(question)
        # For the final app, we still want to show sources
        sources = self.retriever(question)
        return {"answer": answer, "sources": sources}

# Test script
if __name__ == '__main__':
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