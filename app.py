import os
import re
import html
import base64
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
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =====================================================
# FIND BACKGROUND IMAGE (AUTO)
# =====================================================
def find_bg():
    candidates = []

    if os.path.isdir(ASSETS_DIR):
        candidates += [
            os.path.join(ASSETS_DIR, f)
            for f in os.listdir(ASSETS_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    if os.path.isdir(DATA_DIR):
        candidates += [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    return candidates[0] if candidates else None


BG_PATH = find_bg()

if BG_PATH:
    with open(BG_PATH, "rb") as f:
        BG_BASE64 = base64.b64encode(f.read()).decode("utf-8")
    BG_CSS = f"url('data:image/png;base64,{BG_BASE64}')"
else:
    BG_CSS = "none"

# =====================================================
# GLOBAL CSS (BACKGROUND IN MIDDLE)
# =====================================================
st.markdown(
    f"""
    <style>
    html, body {{
        background: #ffffff !important;
    }}

    .stApp {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    /* BACKGROUND IMAGE LAYER */
    body::before {{
        content: "";
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 1200px;
        max-width: 95vw;
        aspect-ratio: 16 / 9;
        background-image: {BG_CSS};
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        z-index: -1;
        pointer-events: none;
        opacity: 1;
    }}

    /* CONTENT CARD */
    section.main > div.block-container {{
        max-width: 980px !important;
        padding: 2.5rem !important;
        margin-top: 10vh !important;
        background: rgba(255,255,255,0.95) !important;
        border-radius: 22px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.18) !important;
        backdrop-filter: blur(8px);
    }}

    .center {{
        text-align: center;
    }}

    .q-bubble {{
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        font-weight: 600;
        margin: 10px 0 8px;
        border: 1px solid rgba(0,0,0,0.06);
        background: #f2f2f2;
    }}

    .answer {{
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
    }}

    div.stButton > button {{
        border-radius: 10px;
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
    if "bayut" in n:
        return "Bayut"
    if "dubizzle" in n:
        return "Dubizzle"
    return "General"

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
# BUILD VECTOR STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()
    stores = {"General": [], "Bayut": [], "Dubizzle": []}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        for q, a in parse_qa_pairs(read_text(fp)):
            doc = Document(page_content=q, metadata={"answer": a})
            stores["General"].append(doc)
            stores[bucket_from_filename(f)].append(doc)

    return {
        k: FAISS.from_documents(v, emb) if v else None
        for k, v in stores.items()
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
    vs = VECTOR_STORES.get(st.session_state.tool_mode)
    if vs:
        results = vs.similarity_search(q, k=4)
        answer = next(
            (r.metadata["answer"] for r in results if r.metadata.get("answer")),
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
    st.markdown(f"<div class='q-bubble'>{html.escape(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
