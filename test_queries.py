import requests
import json

url = "http://localhost:8000/query"
questions = [
    "How does Shor's algorithm threaten current cryptographic systems, and what are some proposed quantum-resistant alternatives?",
    "What evidence supports the connection between gut microbiota composition and the severity of depression symptoms?",
    "Trace the evolution of the mandolin in Italian music from its origins to its role in modern compositions.",
    "What are the primary environmental concerns associated with deep-sea mining, and how might they be mitigated?",
    "Describe a specific application of CRISPR technology in agriculture, including the target crop and the genetic modification made."
]

for question in questions:
    response = requests.post(url, json={"query": question})
    resp_data = response.json()
    
    print(f"Question: {question}")
    print(f"Answer: {resp_data['answer']}")
    
    # Display source information if available
    if 'sources' in resp_data and resp_data['sources']:
        print("\nSources used:")
        for i, source in enumerate(resp_data['sources']):
            print(f"  Source {i+1}: {source['filename']}")
            print(f"  Preview: {source['preview'][:100]}...")
    elif 'used_knowledge_base' in resp_data:
        if not resp_data['used_knowledge_base']:
            print("\nNo relevant sources found in knowledge base.")
    
    print("\n" + "-"*80 + "\n")