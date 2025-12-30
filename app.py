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
# FIND HERO IMAGE (AUTO)
# =====================================================
def find_hero_image():
    # Prefer assets, then data, then root
    for folder in [ASSETS_DIR, DATA_DIR, BASE_DIR]:
        if os.path.isdir(folder):
            imgs = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                # pick the newest file so when you re-upload it changes
                imgs_full = [os.path.join(folder, f) for f in imgs]
                imgs_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return imgs_full[0]
    return None

HERO_PATH = find_hero_image()
HERO_B64 = ""

if HERO_PATH and os.path.isfile(HERO_PATH):
    with open(HERO_PATH, "rb") as f:
        HERO_B64 = base64.b64encode(f.read()).decode("utf-8")

# =====================================================
# GLOBAL CSS (BIGGER HERO)
# =====================================================
# Key changes:
# - hero width: 98vw (almost full screen)
# - hero height: 520px (bigger)
# - background-size: cover (fills the hero area)
# - content padding-top increased so UI starts after hero
st.markdown(
    f"""
    <style>
    html, body {{
        background: #ffffff !important;
    }}

    .stApp {{
        background: #ffffff !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: #ffffff !important;
    }}

    /* BIG HERO IMAGE */
    .hero {{
        position: fixed;
        top: 0px;
        left: 50%;
        transform: translateX(-50%);
        width: 98vw;
        max-width: 1600px;
        height: 520px;
        background-image: url("data:image/png;base64,{HERO_B64}");
        background-repeat: no-repeat;
        background-size: cover;
        background-position: center center;
        opacity: 0.32;
        z-index: 0;
        pointer-events: none;
    }}

    /* MAIN CONTENT WRAPPER */
    section.main > div.block-container {{
        position: relative;
        z-index: 2;
        max-width: 980px !important;
        padding-top: 6.2rem !important;
        padding-bottom: 3rem !important;
        background: transparent !important;
    }}

    /* push UI down so it starts BELOW the hero */
    .push-down {{
        height: 280px;
    }}

    /* CARD */
    .ui-card {{
        background: rgba(255,255,255,0.96);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.08);
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
        background: #f2f2f2;
        border: 1px solid rgba(0,0,0,0.05);
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

    <div class="hero"></div>
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
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD VECTOR STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()
    stores = {"General": [], "Bayut": [], "Dubizzle": []}

    if not os.path.isdir(DATA_DIR):
        return {k: None for k in stores}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        for q, a in parse_qa_pairs(read_text(fp)):
            doc = Document(page_content=q, metadata={"answer": a})
            stores["General"].append(doc)
            stores[bucket_from_filename(f)].append(doc)

    return {k: (FAISS.from_documents(v, emb) if v else None) for k, v in stores.items()}

VECTOR_STORES = build_stores()

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='push-down'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <h1 class="center" style="margin-bottom:6px;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span> AI Content Assistant
    </h1>
    <p class="center" style="margin-top:0;">Internal AI Assistant</p>
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
    f"<h3 class='center' style='margin-top:18px;'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# UI CARD START
# =====================================================
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)

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

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": answer})

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{html.escape(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)
