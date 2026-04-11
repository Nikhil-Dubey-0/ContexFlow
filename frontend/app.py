import sys
sys.path.insert(0, ".")  # so we can import from app/

import streamlit as st
from app.services.rag_pipeline import RAGPipeline
from app.ingestion.embedding_pipeline import IngestionPipeline
import os
import shutil


# --- Page config (first Streamlit command) ---
st.set_page_config(
    page_title="ContexFlow",
    page_icon="🔍",
    layout="wide"
)

# --- Load pipeline ONCE using Streamlit's caching ---
# @st.cache_resource means: run this function once, then reuse the result
# without this, the pipeline would reload on every user interaction
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()


# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True  # allow multiple files at once
    )
    
    if uploaded_files and st.button("📥 Ingest Documents"):
        # save uploaded files to data/raw/
        for file in uploaded_files:
            save_path = os.path.join("data", "raw", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Saved: {file.name}")
        
        # run ingestion pipeline
        with st.spinner("Processing documents... This may take a minute."):
            ingestion = IngestionPipeline()
            ingestion.run()
        
        st.success("✅ Documents ingested! You can now ask questions.")
        # clear the cached pipeline so it reloads with new data
        st.cache_resource.clear()
    
    st.divider()
    st.caption("ContexFlow v1.0 — RAG System")


# --- Main area: Chat Interface ---
st.title("🔍 ContexFlow")
st.caption("Ask questions about your documents")

# session_state keeps data alive across user interactions
# without this, chat history would disappear on every rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

# display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.markdown(message["content"])
        
        # if assistant message has sources, show them
        if "sources" in message:
            with st.expander("📚 Sources"):
                for s in message["sources"]:
                    st.write(f"**{s['source']}** — Page {s['page']}")
                    st.caption(s["snippet"][:200])
                    st.divider()

# chat input box at the bottom
if prompt := st.chat_input("Ask a question about your documents..."):
    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.query(prompt, top_k=5)
        
        # display the answer
        st.markdown(result["answer"])
        
        # display sources in an expandable section
        with st.expander("📚 Sources"):
            for s in result["sources"]:
                st.write(f"**{s['source']}** — Page {s['page']}")
                st.caption(s["snippet"][:200])
                st.divider()
    
    # save assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
