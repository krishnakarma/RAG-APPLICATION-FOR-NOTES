from pathlib import Path
from langchain_community.document_loaders import TextLoader

file_path = Path(__file__).parent / "notes.txt"

data = TextLoader(str(file_path))

docs = data.load()

print(docs[0].page_content)