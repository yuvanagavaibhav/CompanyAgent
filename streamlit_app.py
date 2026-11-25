import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm, extract_text

from src.config import DATA_DIR, INDEX_DIR, TOP_K
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
    
    # API Key check (Google Gemini or Groq)
    has_google = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    if not (has_google or has_groq):
        gemini_input = st.text_input("Enter Google / Gemini API Key:", type="password")
        if gemini_input:
            os.environ["GOOGLE_API_KEY"] = gemini_input
            has_google = True
        else:
            groq_input = st.text_input("Or Enter Groq API Key:", type="password")
            if groq_input:
                os.environ["GROQ_API_KEY"] = groq_input
                has_groq = True
            else:
                st.warning("⚠️ No API key found. Add `GOOGLE_API_KEY` to `.env` or enter above.")

    st.subheader("⚙️ Settings")
    company_name = st.text_input("Company / Organization Name:", value="Setu Tech")
    active_provider = "Google Gemini" if has_google else ("Groq" if has_groq else "None")
    st.info(f"**Active Provider:** {active_provider}")
    selected_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=10, value=TOP_K)
    
    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    
    # Dynamic PDF Upload Section
    with st.expander("📥 Upload New Documents", expanded=False):
        replace_all = st.checkbox(
            "Wipe existing docs (Switch Company)", 
            value=False,
            help="Check this if you are onboarding a new company and want to remove previous PDFs."
        )
        uploaded_files = st.file_uploader(
            "Choose PDF files from local machine",
            type=["pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("⚡ Ingest & Index Uploaded PDFs", use_container_width=True):
                with st.spinner("Saving and indexing documents..."):
                    if replace_all:
                        # Clear old PDFs
                        for old_pdf in DATA_DIR.glob("*.pdf"):
                            old_pdf.unlink()
                        # Clear index directory
                        if INDEX_DIR.exists():
                            for idx_file in INDEX_DIR.glob("*"):
                                idx_file.unlink(missing_ok=True)

                    # Save uploaded PDFs to data directory
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    for uf in uploaded_files:
                        save_dest = DATA_DIR / uf.name
                        with open(save_dest, "wb") as f:
                            f.write(uf.getbuffer())

                    # Rebuild the FAISS index with the new files
                    rebuild_index()
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": f"🎉 Knowledge base updated with {len(uploaded_files)} new document(s)! Ask me anything about them."
                        }
                    ]
                st.success("Successfully ingested and indexed!")
                st.rerun()

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    st.write(f"**Active Documents ({len(pdf_files)}):**")
    for f in pdf_files:
        st.caption(f"📄 {f.name}")
        
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
st.title(f"💼 {company_name} Knowledge Assistant")
st.markdown(f"Ask natural language questions grounded in verified **{company_name}** documents.")

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
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
        with st.chat_message("assistant"):
            st.error("Please provide a valid `GOOGLE_API_KEY` or `GROQ_API_KEY` in `.env` or in the sidebar.")
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

    # 4. Generate Answer
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are CompanyAgent, a helpful and precise internal assistant for {company_name}.
Answer the user's question based strictly on the provided company documents.

CONTEXT:
{{context}}

RULES:
1. Answer directly and factually using only the provided context.
2. If the context does not contain enough information to answer the question, clearly respond:
   "I don't know based on the provided company documents."
3. Do not speculate or make up policies or procedures.
4. When relevant, cite the specific policy, section, or document mentioned.
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
                llm = get_llm(temperature=0.1)
                response = llm.invoke(messages)
                raw_answer = extract_text(response.content)
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
