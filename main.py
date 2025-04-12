from fastapi import FastAPI, UploadFile, File, HTTPException
from pdfminer.high_level import extract_text
import numpy as np
import requests
import os
import re
import time
from typing import List, Dict

# Initialize FastAPI app and global variables
app = FastAPI()
chunks = []  # In-memory storage for chunks
api_key = "WCDPibbdfjYjRDCiIjNpiOBqco6Uie40"
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable is not set")
embedding_url = "https://api.mistral.ai/v1/embeddings"
chat_url = "https://api.mistral.ai/v1/chat/completions"

# Ingestion endpoint
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Ingest PDF files, extract text, and store chunks with embeddings using batch processing."""
    processed_count = 0
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        try:
            text = extract_text(file.file)
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Filter out empty paragraphs
            valid_paragraphs = [para for para in paragraphs if para.strip()]
            
            print(f"Processing {len(valid_paragraphs)} paragraphs from {file.filename}")
            
            # Process paragraphs in batches
            batch_size = 10  # Adjust based on your API limits
            
            for i in range(0, len(valid_paragraphs), batch_size):
                batch_paragraphs = valid_paragraphs[i:i + batch_size]
                
                # Get embeddings for the whole batch at once
                batch_embeddings = get_embeddings_batch(batch_paragraphs, batch_size)
                
                # Add each paragraph and its embedding to chunks
                for j, (para, embedding) in enumerate(zip(batch_paragraphs, batch_embeddings)):
                    chunks.append({
                        "text": para,
                        "embedding": embedding,
                        "filename": file.filename,
                        "index": i + j
                    })
                    processed_count += 1
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
    
    return {"status": "ingested", "processed_chunks": processed_count}

# Query endpoint
@app.post("/query")
async def query(data: dict):
    """Process a user query and return an answer."""
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
        
    # Debug information about the knowledge base
    print(f"\nKnowledge base status:")
    print(f"Total chunks in memory: {len(chunks)}")
    if chunks:
        print(f"Sample chunk filenames: {[chunk['filename'] for chunk in chunks[:3]]}")
    else:
        print("WARNING: No chunks in knowledge base!")
        
    # Check if the query needs a search
    should_search = needs_search(query)
    if not should_search:
        print("Query doesn't need search according to heuristics")
        return {
            "answer": "Hello! How can I assist you?",
            "used_knowledge_base": False,
            "sources": []
        }
    
    print(f"\nProcessing query: {query}")
    query_embedding = get_embedding(query)
    
    print("Running semantic search...")
    semantic_results = semantic_search(query_embedding, 5)
    print(f"Semantic search returned {len(semantic_results)} results")
    
    print("Running keyword search...")
    keyword_results = keyword_search(query, 5)
    print(f"Keyword search returned {len(keyword_results)} results")
    
    print("Post-processing results...")
    top_chunks = post_process(semantic_results, keyword_results, query)
    
    # Log information about the top chunks
    print(f"Top {len(top_chunks)} chunks retrieved:")
    for i, chunk in enumerate(top_chunks):
        print(f"  Chunk {i+1}: from {chunk['filename']}, {len(chunk['text'])} chars")
        # Print a preview of the chunk text (first 100 chars)
        preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
        print(f"  Preview: {preview}")
    
    print("Generating answer...")
    answer = generate_answer(query, top_chunks)
    
    # Return top chunks along with the answer
    return {
        "answer": answer,
        "used_knowledge_base": bool(top_chunks),
        "sources": [
            {
                "filename": chunk["filename"],
                "preview": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"]
            } for chunk in top_chunks
        ]
    }

# Helper functions
def get_embeddings_batch(texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
    """Get embeddings for multiple texts in batches using Mistral AI API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    all_embeddings = []
    
    # Process in batches to avoid overloading the API
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        max_retries = 3
        base_delay = 1  # seconds
        
        for retry in range(max_retries):
            try:
                payload = {"input": batch, "model": "mistral-embed"}
                response = requests.post(embedding_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    batch_embeddings = [np.array(item["embedding"]) for item in response.json()["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                # Handle rate limiting
                if response.status_code == 429 and retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)
                    print(f"Rate limit exceeded. Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
                    continue
                
                # For other errors
                error_msg = f"Failed to get embeddings batch: Status {response.status_code}, Response: {response.text}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
                
            except Exception as e:
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)
                    print(f"Exception in batch. Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
                else:
                    error_msg = f"Failed to get embeddings after {max_retries} retries: {str(e)}"
                    print(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
        
        # Small delay between batches, but much smaller now that rate limits are higher
        if i + batch_size < len(texts):
            time.sleep(0.1)
    
    return all_embeddings

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a single text using batch function."""
    return get_embeddings_batch([text])[0]

def needs_search(query: str) -> bool:
    """Determine if the query requires a search based on enhanced heuristics."""
    query_lower = query.lower().strip()
    
    print(f"Evaluating if query needs search: '{query}'")
    
    # Expanded question words and request verbs
    question_words = [
        "what", "how", "why", "when", "where", "who", "is", "are", "can", "could", "would",
        "tell", "explain", "describe", "show", "list", "find", "search"
    ]
    
    # Check for question words in the query
    if any(word in query_lower.split() for word in question_words):
        print("  - Search needed: Contains question word")
        return True
    
    # Check for question mark
    if query.endswith("?"):
        print("  - Search needed: Ends with question mark")
        return True
    
    # Check for imperative statements or request phrases
    request_phrases = [
        "tell me", "explain", "describe", "i want to know", "i'm curious about", "can you tell me",
        "what is", "how does", "why is", "when did", "where is", "who is"
    ]
    if any(query_lower.startswith(phrase) for phrase in request_phrases):
        print("  - Search needed: Starts with request phrase")
        return True
    
    # Use regex to detect specific patterns
    patterns = [
        r"^(what|how|why|when|where|who|can|could|would|tell|explain|describe)",  # Starts with key words
        r"\b(can|could|would)\b"  # Contains modal verbs
    ]
    for pattern in patterns:
        if re.search(pattern, query_lower):
            print(f"  - Search needed: Matches pattern {pattern}")
            return True
    
    # If no conditions are met, assume no search is needed
    print("  - No search needed: Query doesn't match any search criteria")
    return False

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