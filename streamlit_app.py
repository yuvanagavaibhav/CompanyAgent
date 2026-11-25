import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import DATA_DIR, INDEX_DIR, GROQ_MODEL, TOP_K
from src.data_ingestion_pipeline import load_pdf, clean_documents, split_documents
from src.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore
from src.retrieval import retrieve_documents
from src.guardrails import validate_input, validate_output

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="CompanyAgent — Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Helper Functions -----------------
@st.cache_resource(show_spinner="Loading vector store index...")
def get_or_create_vectorstore():
    """Load existing FAISS index or build it if missing."""
    index_file = INDEX_DIR / "index.faiss"
    if index_file.exists():
        try:
            return load_vectorstore(INDEX_DIR)
        except Exception as e:
            st.warning(f"Could not load existing index ({e}). Rebuilding...")
    
    # Build if missing
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        return None
    raw_docs = load_pdf(pdf_paths)
    cleaned = clean_documents(raw_docs)
    chunks = split_documents(cleaned)
    store = build_vectorstore(chunks)
    save_vectorstore(store, INDEX_DIR)
    return store

def rebuild_index():
    """Forces rebuild of the FAISS index from the PDFs."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        st.error("No PDF files found in data/ directory!")
        return None
    with st.spinner("Processing documents and building FAISS index..."):
        raw_docs = load_pdf(pdf_paths)
        cleaned = clean_documents(raw_docs)
        chunks = split_documents(cleaned)
        store = build_vectorstore(chunks)
        save_vectorstore(store, INDEX_DIR)
    st.cache_resource.clear()
    return store

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("🏢 CompanyAgent")
    st.caption("AI-powered Knowledge & Policy Assistant")
    st.markdown("---")
    
    # API Key check
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        api_key_input = st.text_input("Enter Groq API Key:", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            groq_api_key = api_key_input
        else:
            st.warning("⚠️ GROQ_API_KEY not found in `.env` or input.")
    
    st.subheader("⚙️ Settings")
    selected_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=10, value=TOP_K)
    st.text(f"Model: {GROQ_MODEL}")
    
    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    st.write(f"**PDFs Found:** {len(pdf_files)}")
    for f in pdf_files:
        st.text(f"• {f.name}")
        
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        rebuild_index()
        st.success("Vector store rebuilt successfully!")
        st.rerun()
        
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- Chat State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **CompanyAgent**. Ask me anything about Setu Tech's workplace policies, engineering runbooks, APIs, or customer support procedures."
        }
    ]

# ----------------- Main Chat Header -----------------
st.title("💼 Setu Tech Knowledge Assistant")
st.markdown("Ask natural language questions grounded in verified company documents.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page']}, Chunk {src['chunk_id']})")
                    st.caption(src["content"])
                    st.divider()

# ----------------- Chat Input & RAG Pipeline -----------------
user_input = st.chat_input("Ask a question (e.g., What is our remote work policy?)...")

if user_input:
    # 1. Display user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Guardrails: Input Validation (Security & PII check)
    is_valid_query, guard_msg = validate_input(user_input)
    if not is_valid_query:
        with st.chat_message("assistant"):
            st.error(f"🛡️ Guardrail Alert: {guard_msg}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🛡️ **Guardrail Alert:** {guard_msg}"
        })
        st.stop()

    # 3. Check prerequisites
    if not os.getenv("GROQ_API_KEY"):
        with st.chat_message("assistant"):
            st.error("Please provide a valid GROQ_API_KEY in `.env` or in the sidebar to generate answers.")
        st.stop()

    vectorstore = get_or_create_vectorstore()
    if not vectorstore:
        with st.chat_message("assistant"):
            st.error("No vector store found or created. Please ensure PDFs exist in `data/`.")
        st.stop()

    # 3. Retrieve chunks
    with st.spinner("Searching company documents..."):
        retrieved_docs = retrieve_documents(vectorstore, user_input, k=selected_k)

    # Format context for prompt
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Prepare sources for UI citation
    sources_data = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "content": doc.page_content
        }
        for doc in retrieved_docs
    ]

    # 4. Generate Answer with Groq
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are CompanyAgent, a helpful and precise internal assistant for Setu Tech.
Answer the user's question based strictly on the provided company documents.

CONTEXT:
{context}

RULES:
1. Answer directly and factually using only the provided context.
2. If the context does not contain enough information to answer the question, clearly respond:
   "I don't know based on the provided company documents."
3. Do not speculate or make up policies or procedures.
4. When relevant, cite the specific policy, severity level, or process mentioned.
"""),
        ("human", "{question}"),
    ])

    messages = rag_prompt.format_messages(
        context=context_text,
        question=user_input
    )

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.1)
                response = llm.invoke(messages)
                raw_answer = response.content
                # Guardrails: Output Validation
                is_valid_out, answer_content = validate_output(raw_answer, context_text)
                if not is_valid_out:
                    answer_content = f"🛡️ **Guardrail Intervention:** {answer_content}"
            except Exception as e:
                answer_content = f"⚠️ An error occurred while generating response: {e}"

        st.markdown(answer_content)
        if sources_data:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(sources_data, 1):
                    st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page']}, Chunk {src['chunk_id']})")
                    st.caption(src["content"])
                    st.divider()

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_content,
        "sources": sources_data
    })
