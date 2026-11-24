import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion_pipeline import load_pdf, clean_documents, split_documents
from src.config import DATA_DIR
from src.vectorstore import build_vectorstore, save_vectorstore
from src.vectorstore import load_vectorstore, get_embeddings
from src.retrieval import retrieve_documents
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

print("total pdf files found:", len(pdf_paths))

documents = load_pdf(pdf_paths)
print("total documents loaded:", len(documents))
# print("before cleaning:", documents[0].page_content[:500])  # Print first 500 characters of the first document
cleaned_documents = clean_documents(documents)
# print("after cleaning:", cleaned_documents[0].page_content[:500])  # Print first 500 characters of the first document

chunks = split_documents(cleaned_documents)
# print("total chunks created:", len(chunks))
# for i,c in enumerate(chunks):
#    print("\n" + "="* 60)
#    print("Chunk:", i)
#    print("source:", c.metadata.get('source'))
#    print("page:", c.metadata.get('page'))
#    print("chunk_id:", c.metadata.get('chunk_id'))
#    print("Length of chunk content:", len(c.page_content))
#    print("First 100 characters of chunk content:", c.page_content[:100])
print("total chunks created:", len(chunks))

vectorstore = build_vectorstore(chunks)
save_vectorstore(vectorstore)

print("Vector store built and saved successfully.")
print("Vector store saved at:", str(Path(".faiss").resolve()))
print("Vector store contains:", vectorstore.index.ntotal, "vectors.")
print("Vector store dimension:", vectorstore.index.d)

save_vectorstore(vectorstore)
print("Vector store built and saved successfully.")

#retrieve documents based on a sample question
question = "What is Setu Tech's remote work and equipment stipend policy?"
print(f"Retrieving documents for the question: '{question}'")
vectorstore = load_vectorstore()

results = retrieve_documents(vectorstore = vectorstore, question = question)

print("Retrieved chunks:", len(results))



rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """ You are a helpful company assistant that provides information based on the provided documents. Don't hallucinate.
    
    RULES:
    1. Always answer based on the provided documents.
    2. If the answer is not present in the documents, respond with 'I don't know' or 'The information is not available in the provided documents.
    3. Do not provide information that is not present in the documents.
    4. If the question is unrelated to the documents, respond with 'I don't know' or 'The information is not available in the provided),
    5. If the question is ambiguous, ask for clarification.

    Retrieved context : {context}

    """),
    ("human", "{question}"),
])

messages = rag_prompt.format_messages(
   context = results,
   question = question
)

llm = ChatGroq(model_name="openai/gpt-oss-20b")
response = llm.invoke(messages)

print("\n" + "="* 60)
print("Question:", question)
print("CompanyAgent says..", response.content)