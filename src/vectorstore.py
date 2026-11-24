from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import EMBEDDING_MODEL, INDEX_DIR

def get_embeddings():
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU available
        encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
    )

def build_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided to build the vector store.")
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        chunks,
        embedding=embeddings
    )
    return vectorstore

def save_vectorstore(store, path = INDEX_DIR):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

def load_vectorstore(path = INDEX_DIR):
    return FAISS.load_local(str(path), embeddings=get_embeddings(), allow_dangerous_deserialization=True)
