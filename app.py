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
# PATHS (ROBUST FOR STREAMLIT CLOUD)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =====================================================
# 🔥 FORCE BACKGROUND IMAGE (NO CACHE / NO GUESSING)
# Put your file in /assets and set its exact name here.
# =====================================================
BG_IMAGE_PATH = os.path.join(ASSETS_DIR, "background.png")  # rename your image to background.png

if os.path.isfile(BG_IMAGE_PATH):
    with open(BG_IMAGE_PATH, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        /* ✅ TOP aligned background */
        html, body, .stApp {{
            background-image: url("data:image/png;base64,{bg_b64}") !important;
            background-repeat: no-repeat !important;
            background-position: top center !important;   /* ✅ START FROM ABOVE */
            background-size: cover !important;
            background-attachment: scroll !important;     /* makes it start at top naturally */
            min-height: 100vh !important;
        }}

        /* Streamlit containers must be transparent so background shows */
        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* Keep content readable (white glass card) */
        section.main > div.block-container {{
            max-width: 980px !important;
            padding: 2rem !important;
            background: rgba(255,255,255,0.92) !important;
            border-radius: 22px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.18) !important;
        }}

        .center {{ text-align:center; }}

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

        div.stButton > button {{
            border-radius: 10px;
        }}

        .small-btn div.stButton > button {{
            padding-top: 0.35rem !important;
            padding-bottom: 0.35rem !important;
            font-size: 0.95rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("❌ assets/background.png NOT FOUND (rename your image to background.png inside /assets)")

# =====================================================
# ACCESS CODE GATE
# =====================================================
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
REQUIRE_CODE = os.getenv("REQUIRE_CODE", "0").strip() == "1"

if REQUIRE_CODE and ACCESS_CODE:
    st.session_state.setdefault("unlocked", False)

    if not st.session_state["unlocked"]:
        st.markdown(
            """
            <div style="text-align:center">
              <h2><span style="color:#0E8A6D;">Bayut</span> &
              <span style="color:#D71920;">Dubizzle</span> AI Assistant</h2>
              <p>Internal AI Assistant – Access Required</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        code = st.text_input("Access code", type="password")

        if st.button("Unlock"):
            if code == ACCESS_CODE:
                st.session_state["unlocked"] = True
                st.rerun()
            else:
                st.error("Wrong access code")

        st.stop()

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("answer_mode", "Ultra-Fast")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# =====================================================
# HELPERS (NO UNICODE ERRORS EVER)
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
# BUILD VECTOR STORES
# =====================================================
@st.cache_resource
def build_stores():
    emb = get_embeddings()
    all_docs, bayut_docs, dub_docs = [], [], []

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ data/ folder not found")

    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        if not os.path.isfile(fp) or f.startswith(".") or is_sop_file(f):
            continue

        text = read_text(fp)
        for q, a in parse_qa_pairs(text):
            doc = Document(page_content=q, metadata={"answer": a, "source": f})
            all_docs.append(doc)

            b = bucket_from_filename(f)
            if b in ("bayut", "both"):
                bayut_docs.append(doc)
            if b in ("dubizzle", "both"):
                dub_docs.append(doc)

    return (
        FAISS.from_documents(all_docs, emb) if all_docs else None,
        FAISS.from_documents(bayut_docs, emb) if bayut_docs else None,
        FAISS.from_documents(dub_docs, emb) if dub_docs else None
    )

VS_ALL, VS_BAYUT, VS_DUBIZZLE = build_stores()

def pick_store():
    return {
        "Bayut": VS_BAYUT,
        "Dubizzle": VS_DUBIZZLE,
        "General": VS_ALL
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
cols = st.columns(3)
if cols[0].button("General", use_container_width=True):
    st.session_state.tool_mode = "General"
if cols[1].button("Bayut", use_container_width=True):
    st.session_state.tool_mode = "Bayut"
if cols[2].button("Dubizzle", use_container_width=True):
    st.session_state.tool_mode = "Dubizzle"

st.markdown(f"<h3 class='center'>{st.session_state.tool_mode} Assistant</h3>", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
q = st.text_input("Type your question here…")
if st.button("Ask") and q:
    vs = pick_store()
    if vs is None:
        st.session_state.chat[st.session_state.tool_mode].append({
            "q": q,
            "a": "No relevant Q&A files found for this section."
        })
    else:
        results = vs.similarity_search(q, k=4)
        answers = []
        for r in results:
            a = r.metadata.get("answer")
            if a and a not in answers:
                answers.append(a)

        st.session_state.chat[st.session_state.tool_mode].append({
            "q": q,
            "a": answers[0] if answers else "No relevant answer found."
        })

# =====================================================
# CHAT
# =====================================================
for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{html.escape(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
