import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =========================================
# STYLE
# =========================================
st.markdown("""
<style>
.block-container{
    max-width:100%;
    padding:2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    mode = st.radio("Mode", ["General", "Bayut", "Dubizzle"])
    answer_mode = st.radio("Answer mode", ["Fast", "Thinking"])

# =========================================
# TITLE
# =========================================
st.markdown("""
<h1 style="text-align:center;">
<span style="color:#0E8A6D;">Bayut</span> &
<span style="color:#D71920;">Dubizzle</span>
AI Content Assistant
</h1>
""", unsafe_allow_html=True)

# =========================================
# PATHS
# =========================================
def get_paths(mode):
    base = "/tmp/faiss"
    if mode == "Bayut":
        return "data/bayut", f"{base}_bayut"
    if mode == "Dubizzle":
        return "data/dubizzle", f"{base}_dubizzle"
    return "data/general", f"{base}_general"

DATA_DIR, INDEX_PATH = get_paths(mode)

# =========================================
# EMBEDDINGS (OPENAI – SAFE)
# =========================================
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# =========================================
# INDEX
# =========================================
@st.cache_resource
def load_or_build_index():
    if not os.path.exists(DATA_DIR):
        return None

    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            docs += TextLoader(os.path.join(DATA_DIR, f), encoding="utf-8").load()

    if not docs:
        return None

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    ).split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index

index = load_or_build_index()

# =========================================
# UI
# =========================================
question = st.text_input(f"Ask a {mode} question")

if st.button("Ask"):
    if not index:
        st.error("No data found.")
    else:
        docs = index.similarity_search(question, k=3)

        if answer_mode == "Fast":
            st.write(docs[0].page_content[:800])
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            context = "\n".join(d.page_content for d in docs)
            st.write(llm.invoke(context + "\n\n" + question).content)
