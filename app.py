import os
import re
import html
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(
    page_title="Bayut & dubizzle AI Content Assistant",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.markdown(
    """
    <style>
      section.main > div.block-container{
        max-width: 980px;
        padding: 2rem;
      }
      .q-bubble{
        padding:10px 14px;
        border-radius:14px;
        margin:10px 0;
        font-weight:600;
      }
      .q-general{background:#f2f2f2}
      .q-bayut{background:#e6f4ef}
      .q-dubizzle{background:#fdeaea}
    </style>
    """,
    unsafe_allow_html=True
)

st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "dubizzle": []})

def is_sop_file(name):
    return "sop" in name.lower()

def bucket_from_filename(name):
    n = name.lower()
    if "bayut" in n:
        return "Bayut"
    if "dubizzle" in n:
        return "dubizzle"
    return "General"

def parse_qa_pairs(text):
    pattern = re.compile(r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)", re.S)
    return pattern.findall(text)

@st.cache_resource
def build_store():
    if not os.path.isdir(DATA_DIR):
        return None
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt") and not is_sop_file(f):
            with open(os.path.join(DATA_DIR, f), "r", encoding="utf-8", errors="ignore") as fh:
                for q, a in parse_qa_pairs(fh.read()):
                    docs.append(Document(page_content=q, metadata={"answer": a}))
    return FAISS.from_documents(docs, emb) if docs else None

VS = build_store()

st.title("Bayut & dubizzle AI Content Assistant")
st.caption("Internal AI Assistant")

cols = st.columns(3)
if cols[0].button("General"): st.session_state.tool_mode = "General"
if cols[1].button("Bayut"): st.session_state.tool_mode = "Bayut"
if cols[2].button("dubizzle"): st.session_state.tool_mode = "dubizzle"

q = st.text_input("Type your question")

if st.button("Ask") and q and VS:
    res = VS.similarity_search(q, k=1)
    ans = res[0].metadata["answer"] if res else "No answer found."
    st.session_state.chat[st.session_state.tool_mode].append((q, ans))
    st.rerun()

for q, a in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{html.escape(q)}</div>", unsafe_allow_html=True)
    st.write(a)
