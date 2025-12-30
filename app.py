import os
import re
import html
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================
# PAGE CONFIG (MUST BE FIRST)
# =====================================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================
# BASIC UI STYLES (NO BACKGROUND IMAGE)
# =====================================================
st.markdown(
    """
    <style>
    .center { text-align: center; }

    section.main > div.block-container {
        max-width: 980px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 2.5rem !important;
    }

    .q-bubble {
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        font-weight: 600;
        margin: 10px 0 8px;
        background: #f2f2f2;
        border: 1px solid rgba(0,0,0,0.06);
    }

    .answer {
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "Dubizzle": []
})

# =====================================================
# HELPERS
# =====================================================
def is_sop_file(name: str) -> bool:
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "bayut" in n or "mybayut" in n:
        return "Bayut"
    if "dubizzle" in n:
        return "Dubizzle"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text: str):
    pattern = re.compile(
        r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)",
        re.S | re.I
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

# =====================================================
# EMBEDDINGS (✅ FIXED)
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================================
# BUILD VECTOR STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()

    buckets = {
        "General": [],
        "Bayut": [],
        "Dubizzle": []
    }

    if not os.path.isdir(DATA_DIR):
        return {k: None for k in buckets}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        bucket = bucket_from_filename(f)

        for q, a in parse_qa_pairs(read_text(fp)):
            doc = Document(page_content=q, metadata={"answer": a})
            buckets["General"].append(doc)
            buckets[bucket].append(doc)

    return {
        k: FAISS.from_documents(v, emb) if v else None
        for k, v in buckets.items()
    }

VECTOR_STORES = build_stores()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 class="center">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span> AI Content Assistant
    </h1>
    <p class="center">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TOOL MODE BUTTONS (UNCHANGED)
# =====================================================
c1, c2, c3 = st.columns(3)

if c1.button("General", use_container_width=True):
    st.session_state.tool_mode = "General"

if c2.button("Bayut", use_container_width=True):
    st.session_state.tool_mode = "Bayut"

if c3.button("Dubizzle", use_container_width=True):
    st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# INPUT
# =====================================================
q = st.text_input("Type your question here…")

if st.button("Ask") and q:
    vs = VECTOR_STORES.get(st.session_state.tool_mode)

    if vs:
        results = vs.similarity_search(q, k=4)
        answer = next(
            (r.metadata.get("answer") for r in results if r.metadata.get("answer")),
            "No relevant answer found."
        )
    else:
        answer = "No data available."

    st.session_state.chat[st.session_state.tool_mode].append({
        "q": q,
        "a": answer
    })

# =====================================================
# CHAT
# =====================================================
for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(
        f"<div class='q-bubble'>{html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='answer'>{item['a']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
