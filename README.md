# RAG Pipeline

This is a simple Retrieval-Augmented Generation (RAG) pipeline built with FastAPI and the Mistral AI API.

## System Design

1. **Data Ingestion**:
   - Endpoint: `/ingest`
   - Accepts PDF files, extracts text using `pdfminer.six`, splits into paragraphs, computes embeddings with Mistral AI, and stores them in memory.

2. **Query Processing**:
   - Endpoint: `/query`
   - Checks if a search is needed (e.g., contains question words or ends with '?').
   - If no search is needed, returns a greeting. Otherwise, performs retrieval and generates an answer.

3. **Semantic Search**:
   - Computes cosine similarity between query and chunk embeddings.
   - Retrieves top-k chunks.

4. **Keyword Search**:
   - Counts matching words between query and chunks.
   - Retrieves top-k chunks.

5. **Post-processing**:
   - Combines semantic and keyword results.
   - Re-ranks using a weighted score (0.7 semantic + 0.3 keyword).

6. **Answer Generation**:
   - Uses Mistral AI to generate an answer based on the top chunks and query.

## Project Structure
rag_pipeline/
├── main.py         # Main FastAPI application
├── pyproject.toml  # Project configuration and dependencies
└── README.md       # Project documentation

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:YMRodriguez/sai-intw.git
   cd sai-intw
2. **Set the Mistral AI API Key**:
Replace your_api_key_here with your actual Mistral AI API key: 
    ```bash
    export MISTRAL_API_KEY="your_api_key_here"
3. **Set Up the Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate

4. **Install Dependencies**:
   ```bash
   uv sync

5. **Run the Server**:
   ```bash
   uvicorn main:app --reload

6. **Use the API**:
   - Ingest PDFs:
     ```bash
     curl -X POST -F "files=@file.pdf" http://localhost:8000/ingest
   - Query the System:
     ```bash
     curl -X POST -d '{"query": "What is X?"}' -H "Content-Type: application/json" http://localhost:8000/query

## Libraries Used

- FastAPI: API framework
- Uvicorn: ASGI server
- pdfminer.six: PDF text extraction
- requests: HTTP requests to Mistral AI API
- numpy: Numerical computations for embeddings

## Notes

- Data is stored in memory and lost on server restart.
- Only PDF files are supported for ingestion.
- Ensure the Mistral AI API key is valid and has sufficient credits.

