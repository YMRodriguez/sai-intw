import requests

# Make a query that should match one of our documents
query = "What are the environmental impacts of deep sea mining?"
response = requests.post(
    "http://localhost:8000/query",
    json={"query": query}
)

# Print the full response for debugging
data = response.json()
print("Response:", data)

# Check if sources were returned
if "sources" in data and data["sources"]:
    print("\nKnowledge base was used! Sources:")
    for i, source in enumerate(data["sources"]):
        print(f"Source {i+1}: {source['filename']}") 