from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (RecursiveCharacterTextSplitter)
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(paths):
    docs = []
    for path in paths:
        path= Path(path)
        loaded = PyPDFLoader(str(path)).load()

        for d in loaded:
            d.metadata['source'] = path.name
            d.metadata['page'] = (
                int(d.metadata.get('page', 0)) + 1
            )
        docs.extend(loaded)
    return docs  

def clean_documents(docs):
    for d in docs:
        d.page_content = " ".join (d.page_content.split())
    return [d for d in docs if d.page_content.strip()]

def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", "", "."]
    )
    chunks = splitter.split_documents(docs)
    for i,c in enumerate(chunks):
        c.metadata['chunk_id'] = i
    return chunks

