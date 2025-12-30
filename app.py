import os
import re
import html
import time
import base64
import hashlib
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
# ✅ CORRECT PATHS (THIS FIXES EVERYTHING)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

BG_IMAGE_PATH = os.path.join(ASSETS_DIR, "ChatGPT Image Dec 30, 2025, 03_14_09 PM.png")

# =====================================================
# ✅ FORCE BACKGROUND IMAGE (NO CACHE)
# =====================================================
if not os.path.isfile(BG_IMAGE_PATH):
    st.error("❌ assets/background.png NOT FOUND")
else:
    with open(BG_IMAGE_PATH, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* FULL PAGE BACKGROUND */
        html, body, .stApp {{
            background-image: url("data:image/png;base64,{bg_b64}") !important;
            background-size: cover !important;
            background-position: center top !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}

        /* REMOVE STREAMLIT DEFAULT BACKGROUND */
        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* CONTENT CARD (TEXT SAFE AREA) */
        section.main > div.block-container {{
            max-width: 980px !important;
            padding: 2.5rem !important;
            background: rgba(255,255,255,0.95) !important;
            border-radius: 24px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2) !important;
            backdrop-filter: blur(8px);
        }}

        .center {{ text-align: center; }}

        .q-bubble {{
            padding: 10px 14px;
            border-radius: 14px;
            max-width: 85%;
            font-weight: 600;
            margin: 10px 0 8px;
        }}

        .q-general {{ background:#f2f2f2; }}
        .q-bayut {{ background:#e6f4ef; }}
        .q-dubizzle {{ background:#fdeaea; }}

        .answer {{
            margin-left: 6px;
            margin-bottom: 14px;
            line-height: 1.6;
        }}

        button {{
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# =====================================================
# HELPERS
# =====================================================
def is_sop_file(name):
    return "sop" in name.lower()

def bucket_from_filename(name):
    n = name.lower()
    if "both" in n:
        return "both"
    if "bayut" in n or "mybayut" in n:
        return "bayut"
    if "dubizzle" in n:
        return "dubizzle"
    return "general"

def read_text(fp):
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text):
    pattern = re.compile(
        r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)",
        re.S | re.I
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================================
# BUILD STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()
    all_docs, bayut_docs, dub_docs = [], [], []

    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        if not os.path.isfile(fp) or is_sop_file(f):
            continue

        for q, a in parse_qa_pairs(read_text(fp)):
            doc = Document(page_content=q, metadata={"answer": a})
            all_docs.append(doc)

            b = bucket_from_filename(f)
            if b in ("bayut", "both"):
                bayut_docs.append(doc)
            if b in ("dubizzle", "both"):
                dub_docs.append(doc)

    return (
        FAISS.from_documents(all_docs, emb),
        FAISS.from_documents(bayut_docs, emb) if bayut_docs else None,
        FAISS.from_documents(dub_docs, emb) if dub_docs else None
    )

VS_ALL, VS_BAYUT, VS_DUBIZZLE = build_stores()

def pick_store():
    return {
        "General": VS_ALL,
        "Bayut": VS_BAYUT,
        "Dubizzle": VS_DUBIZZLE
    }[st.session_state.tool_mode]

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
# TOOL MODE
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
    vs = pick_store()
    results = vs.similarity_search(q, k=4)

    answer = next(
        (r.metadata["answer"] for r in results if r.metadata.get("answer")),
        "No relevant answer found."
    )

    st.session_state.chat[st.session_state.tool_mode].append({
        "q": q,
        "a": answer
    })

# =====================================================
# CHAT
# =====================================================
for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{html.escape(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
