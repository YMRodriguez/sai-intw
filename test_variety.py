import requests
import json

url = "http://localhost:8000/query"

# Test both types of queries
queries = [
    # Should trigger search
    "What are the environmental impacts of deep sea mining?",
    "How does quantum cryptography protect data?",
    "Tell me about the mandolin in Italian music",
    
    # Should NOT trigger search
    "Hello there",
    "Good morning",
    "I am looking forward to our meeting"
]

for query in queries:
    print("\n" + "="*60)
    print(f"Testing query: \"{query}\"")
    
    response = requests.post(url, json={"query": query})
    data = response.json()
    
    # Print basic info
    print(f"Answer starts with: \"{data['answer'][:100]}...\"")
    
    # Check if knowledge base was used
    if 'used_knowledge_base' in data:
        if data['used_knowledge_base']:
            print("USED KNOWLEDGE BASE: Yes")
            print(f"Sources: {len(data.get('sources', []))} found")
            for i, source in enumerate(data.get('sources', [])):
                print(f"  - {source['filename']}")
        else:
            print("USED KNOWLEDGE BASE: No")
    else:
        print("Response doesn't indicate if knowledge base was used") 