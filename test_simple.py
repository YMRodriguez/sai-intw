import requests
import json

url = "http://localhost:8000/query"

# Try a question that should have an answer in our knowledge base
question = "What are the primary environmental concerns associated with deep-sea mining?"

response = requests.post(url, json={"query": question})
resp_data = response.json()

# Print the full response data for debugging
print("\nFull response data:")
print(json.dumps(resp_data, indent=2))

print(f"\nQuestion: {question}")
print(f"Answer: {resp_data['answer']}")

# Display source information
print("\nSources used:")
if 'sources' in resp_data and resp_data['sources']:
    for i, source in enumerate(resp_data['sources']):
        print(f"  Source {i+1}: {source['filename']}")
        print(f"  Preview: {source['preview'][:100]}...")
else:
    print("  No source information returned") 