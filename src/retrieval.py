from .config import TOP_K

def retrieve_documents(vectorstore, question, k=TOP_K):
    results = vectorstore.similarity_search(query = question, k=k)
    return results