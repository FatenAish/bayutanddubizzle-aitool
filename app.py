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
# BACKGROUND IMAGE (STATIC & SAFE)
# =====================================================
def get_bg_image():
    for folder in [ASSETS_DIR, DATA_DIR]:
        if os.path.isdir(folder):
            imgs = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                return os.path.join(folder, imgs[0])
    return None

BG_PATH = get_bg_image()
BG_B64 = ""

if BG_PATH:
    with open(BG_PATH, "rb") as f:
        BG_B64 = base64.b64encode(f.read()).decode("utf-8")

# =====================================================
# CSS — DO NOT MOVE UI
# =====================================================
st.markdown(
    f"""
    <style>
    /* DO NOT TOUCH LAYOUT */
    html, body {{
        background: white !important;
    }}

    /* BACKGROUND IMAGE ONLY */
    .stApp {{
        background-image: url("data:image/png;base64,{BG_B64}");
        background-repeat: no-repeat;
        background-position: center 120px;
        background-size: 1400px auto;
    }}

    /* KEEP DEFAULT STREAMLIT BEHAVIOR */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* CONTENT CARD (OPTIONAL SOFT BACKDROP) */
    section.main > div.block-container {{
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.06);
    }}

    .center {{ text-align: center; }}

    .q-bubble {{
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        font-weight: 600;
        margin: 10px 0 8px;
        background: #f2f2f2;
    }}

    .answer {{
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
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
    if "bayut" in name.lower():
        return "Bayut"
    if "dubizzle" in name.lower():
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
    return HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# VECTOR STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()
    stores = {"General": [], "Bayut": [], "Dubizzle": []}

    if not os.path.isdir(DATA_DIR):
        return {k: None for k in stores}

    for f in os.listdir(DATA_DIR):
        if not f.endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        for q, a in parse_qa_pairs(read_text(fp)):
            doc = Document(page_content=q, metadata={"answer": a})
            stores["General"].append(doc)
            stores[bucket_from_filename(f)].append(doc)

    return {k: (FAISS.from_documents(v, emb) if v else None) for k, v in stores.items()}

VECTOR_STORES = build_stores()

# =====================================================
# HEADER (UNCHANGED)
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
# TOOL MODE (UNCHANGED)
# =====================================================
c1, c2, c3 = st.columns(3)
if c1.button("General", use_container_width=True):
    st.session_state.tool_mode = "General"
if c2.button("Bayut", use_container_width=True):
    st.session_state.tool_mode = "Bayut"
if c3.button("Dubizzle", use_container_width=True):
    st.session_state.tool_mode = "Dubizzle"

st.markdown(f"<h3 class='center'>{st.session_state.tool_mode} Assistant</h3>", unsafe_allow_html=True)

# =====================================================
# INPUT (UNCHANGED)
# =====================================================
q = st.text_input("Type your question here…")
if st.button("Ask") and q:
    vs = VECTOR_STORES.get(st.session_state.tool_mode)
    answer = "No data available."
    if vs:
        res = vs.similarity_search(q, k=4)
        if res:
            answer = res[0].metadata.get("answer", answer)

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": answer})

# =====================================================
# CHAT (UNCHANGED)
# =====================================================
for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{html.escape(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
