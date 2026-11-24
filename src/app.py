from src.data_ingestion_pipeline import load_pdf
from src.config import DATA_DIR

pdf_paths = sorted (DATA_DIR.glob("*.pdf"))

print("total pdf files found:", len(pdf_paths))

documents=load_pdf(pdf_paths)