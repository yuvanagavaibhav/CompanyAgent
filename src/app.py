import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion_pipeline import load_pdf,clean_documents
from src.config import DATA_DIR

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

print("total pdf files found:", len(pdf_paths))

documents = load_pdf(pdf_paths)
print("total documents loaded:", len(documents))
print("before cleaning:", documents[0].page_content[:500])  # Print first 500 characters of the first document
cleaned_documents = clean_documents(documents)
print("after cleaning:", cleaned_documents[0].page_content[:500])  # Print first 500 characters of the first document