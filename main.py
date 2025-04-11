from fastapi import FastAPI, UploadFile, File, HTTPException
from pdfminer.high_level import extract_text
import numpy as np
import requests
import os
import re
from typing import List, Dict

# Initialize FastAPI app and global variables
app = FastAPI()
chunks = []  # In-memory storage for chunks
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable is not set")
embedding_url = "https://api.mistral.ai/v1/embeddings"
chat_url = "https://api.mistral.ai/v1/chat/completions"

# Ingestion endpoint
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Ingest PDF files, extract text, and store chunks with embeddings."""
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        text = extract_text(file.file)
        paragraphs = re.split(r'\n\s*\n', text)
        for i, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                embedding = get_embedding(para)
                chunks.append({
                    "text": para,
                    "embedding": embedding,
                    "filename": file.filename,
                    "index": i
                })
    return {"status": "ingested"}

# Query endpoint
@app.post("/query")
async def query(data: dict):
    """Process a user query and return an answer."""
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not needs_search(query):
        return {"answer": "Hello! How can I assist you?"}
    
    query_embedding = get_embedding(query)
    semantic_results = semantic_search(query_embedding, 5)
    keyword_results = keyword_search(query, 5)
    top_chunks = post_process(semantic_results, keyword_results, query)
    answer = generate_answer(query, top_chunks)
    return {"answer": answer}

# Helper functions
def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a given text using Mistral AI API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(embedding_url, json={"input": text, "model": "mistral-embed"}, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get embedding")
    return np.array(response.json()["data"][0]["embedding"])

def needs_search(query: str) -> bool:
    """Determine if the query requires a search."""
    question_words = ["what", "how", "why", "when", "where", "who", "is", "are"]
    return any(word in query.lower() for word in question_words) or query.endswith("?")

def semantic_search(query_embedding: np.ndarray, top_k: int) -> List[Dict]:
    """Perform semantic search using cosine similarity."""
    if not chunks:
        return []
    embeddings = np.array([chunk["embedding"] for chunk in chunks])
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def keyword_search(query: str, top_k: int) -> List[Dict]:
    """Perform keyword-based search."""
    if not chunks:
        return []
    query_words = set(re.findall(r'\w+', query.lower()))
    scores = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
        score = len(query_words.intersection(chunk_words))
        scores.append(score)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices if scores[i] > 0]

def post_process(semantic_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
    """Combine and re-rank semantic and keyword search results."""
    combined = {chunk["text"]: chunk for chunk in semantic_results + keyword_results}.values()
    query_embedding = get_embedding(query)
    scores = []
    for chunk in combined:
        semantic_score = np.dot(chunk["embedding"], query_embedding) / (np.linalg.norm(chunk["embedding"]) * np.linalg.norm(query_embedding))
        keyword_score = len(set(re.findall(r'\w+', query.lower())).intersection(set(re.findall(r'\w+', chunk["text"].lower())))) / len(set(re.findall(r'\w+', query.lower())))
        combined_score = 0.7 * semantic_score + 0.3 * keyword_score
        scores.append((chunk, combined_score))
    sorted_chunks = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return [chunk for chunk, _ in sorted_chunks]

def generate_answer(query: str, top_chunks: List[Dict]) -> str:
    """Generate an answer using Mistral AI based on top chunks."""
    context = "\n".join([chunk["text"] for chunk in top_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(chat_url, json={
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": prompt}]
    }, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate answer")
    return response.json()["choices"][0]["message"]["content"]