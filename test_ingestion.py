import requests
import os

url = "http://localhost:8000/ingest"
pdf_dir = "pdf_testing_docs"
files = [
    ("files", (f, open(os.path.join(pdf_dir, f), "rb")))
    for f in os.listdir(pdf_dir) if f.endswith(".pdf")
]
response = requests.post(url, files=files)
print(response.json())