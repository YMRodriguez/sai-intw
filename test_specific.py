import requests
import json

url = "http://localhost:8000/query"

# Try a specific question that should trigger search
question = "What specific damage can deep-sea mining cause to marine habitats?"

print(f"Sending query: {question}")
response = requests.post(url, json={"query": question})
resp_data = response.json()

# Print the full response data
print("\nFull response data:")
print(json.dumps(resp_data, indent=2))

# Check source information
if 'sources' in resp_data and resp_data['sources']:
    print("\nKnowledge base was used! Sources found:")
    for i, source in enumerate(resp_data['sources']):
        print(f"Source {i+1}: {source['filename']}")
else:
    print("\nNo sources found in the response") 