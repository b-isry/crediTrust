# CreditRust RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for exploring and searching consumer complaint narratives using semantic search and vector databases.

## Features

- Cleans and preprocesses consumer complaint data
- Chunks narratives and generates embeddings using Sentence Transformers
- Stores and indexes data in ChromaDB for fast semantic search
- Modern Gradio UI for interactive Q&A
- Ready for integration with chatbots or search applications

## Project Structure

```
creditrust-rag-chatbot/
├── data/
│   └── complaints.csv
├── notebooks/
│   └── 01-EDA-and-Preprocessing.ipynb
├── src/
│   └── vector_store_manager.py
├── app.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Prepare the data**

   - Place your `complaints.csv` in the `data/` directory.
   - Run the notebook `notebooks/01-EDA-and-Preprocessing.ipynb` to generate `filtered_complaints.csv`.

## Building the Vector Store

Run the following command to build the vector store:

```powershell
python src/vector_store_manager.py
```

This will process the data, generate embeddings, and store them in the `vector_store/` directory.

## Running the Gradio App

To launch the chatbot UI:

```powershell
python app.py
```

## CI/CD

- Automated with GitHub Actions (`.github/workflows/ci.yml`)
- Installs dependencies, lints code, runs tests, and performs a Gradio app launch smoke test
- Includes a deploy placeholder

## Requirements

- Python 3.11+
- See `requirements.txt` for Python package dependencies

## License

MIT License
