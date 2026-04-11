import sys
sys.path.insert(0, ".")

import streamlit as st
from app.services.rag_pipeline import RAGPipeline
from app.ingestion.embedding_pipeline import IngestionPipeline
import os


# --- Page config (must be first Streamlit command) ---
st.set_page_config(
    page_title="ContexFlow — Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    /* main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
    }
    /* source cards */
    .source-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        border-left: 3px solid #667eea;
    }
    /* sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
    /* chat input styling */
    .stChatInput {
        border-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load pipeline ONCE using Streamlit's caching ---
@st.cache_resource
def load_pipeline():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = load_pipeline()


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📄 Document Manager")
    
    # show current document count
    if pipeline and hasattr(pipeline.retriever, 'documents'):
        doc_count = len(pipeline.retriever.documents)
        st.metric("Indexed Chunks", doc_count)
    
    st.divider()
    
    # file upload
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📥 Ingest Documents", use_container_width=True):
        os.makedirs("data/raw", exist_ok=True)
        for file in uploaded_files:
            save_path = os.path.join("data", "raw", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"✅ {file.name}")
        
        with st.spinner("🔄 Processing documents..."):
            try:
                ingestion = IngestionPipeline()
                ingestion.run()
                st.success("✅ Documents ingested!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
    
    st.divider()
    
    # clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("ContexFlow v1.0")
    st.caption("Built with FAISS • BM25 • Groq")


# --- Main Chat Area ---
st.markdown('<p class="main-header">ContexFlow</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents — powered by hybrid RAG</p>', unsafe_allow_html=True)
st.markdown("")

# show hint if no documents indexed yet
if pipeline and hasattr(pipeline.retriever, 'documents') and len(pipeline.retriever.documents) == 0:
    st.info("👋 No documents indexed yet. Upload PDFs or DOCX files using the sidebar to get started!")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources", expanded=False):
                for s in message["sources"]:
                    st.markdown(f"**📄 {s['source']}** — Page {s['page']}")
                    st.caption(s["snippet"][:200])
                    st.divider()

# chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not pipeline:
        st.error("Pipeline not loaded. Check your configuration.")
    else:
        # add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # generate streaming response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching documents..."):
                    token_stream, sources, rewritten_query = pipeline.stream_query(
                        prompt, top_k=5, chat_history=st.session_state.messages
                    )
                
                answer = st.write_stream(token_stream)
                
                if sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for s in sources:
                            st.markdown(f"**📄 {s['source']}** — Page {s['page']}")
                            st.caption(s["snippet"][:200])
                            st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "sources": []
                })
