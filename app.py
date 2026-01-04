import os
import re
import html
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================
# PAGE CONFIG
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
# UI STYLES (NO BACKGROUND – SAFE)
# =====================================================
st.markdown(
    """
    <style>
      .center { text-align:center; }

      section.main > div.block-container{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
      }

      .q-bubble{
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        width: fit-content;
        font-weight: 600;
        margin: 10px 0 8px;
        border: 1px solid rgba(0,0,0,0.06);
      }
      .q-general { background:#f2f2f2; }
      .q-bayut { background:#e6f4ef; }
      .q-dubizzle { background:#fdeaea; }

      .answer{
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
st.session_state.setdefault("answer_mode", "Ultra-Fast")
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
        r"Q[:\\-]\\s*(.*?)\\nA[:\\-]\\s*(.*?)(?=\\nQ[:\\-]|\\Z)",
        re.S | re.I
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def format_thinking_answer(primary, extras):
    out = [primary] + extras
    cleaned = []
    for x in out:
        if x and x not in cleaned:
            cleaned.append(x)
    return "\\n\\n".join(cleaned[:4])

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
    if not os.path.isdir(DATA_DIR):
        return None, None, None

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

    return (
        FAISS.from_documents(stores["General"], emb) if stores["General"] else None,
        FAISS.from_documents(stores["Bayut"], emb) if stores["Bayut"] else None,
        FAISS.from_documents(stores["Dubizzle"], emb) if stores["Dubizzle"] else None,
    )

VS_ALL, VS_BAYUT, VS_DUBIZZLE = build_stores()

def pick_store():
    return {
        "General": VS_ALL,
        "Bayut": VS_BAYUT,
        "Dubizzle": VS_DUBIZZLE
    }[st.session_state.tool_mode]

# =====================================================
# HEADER (LOCK ICON RETURNS HERE ✅)
# =====================================================
st.title("Bayut & Dubizzle AI Content Assistant")
st.caption("Internal AI Assistant")

# =====================================================
# TOOL MODE BUTTONS
# =====================================================
cols = st.columns(3)
if cols[0].button("General", use_container_width=True):
    st.session_state.tool_mode = "General"
if cols[1].button("Bayut", use_container_width=True):
    st.session_state.tool_mode = "Bayut"
if cols[2].button("Dubizzle", use_container_width=True):
    st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE
# =====================================================
mode_cols = st.columns(2)
if mode_cols[0].button("Ultra-Fast", use_container_width=True):
    st.session_state.answer_mode = "Ultra-Fast"
if mode_cols[1].button("Thinking", use_container_width=True):
    st.session_state.answer_mode = "Thinking"

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
q = st.text_input("Type your question here…")

btn_cols = st.columns(2)
ask = btn_cols[0].button("Ask", use_container_width=True)
clear = btn_cols[1].button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWER
# =====================================================
if ask and q:
    vs = pick_store()
    if vs is None:
        final = "No internal Q&A data found."
    else:
        thinking = st.session_state.answer_mode == "Thinking"
        if thinking:
            with st.spinner("Thinking…"):
                time.sleep(0.3)

        results = vs.similarity_search(q, k=8 if thinking else 4)
        answers = [r.metadata["answer"] for r in results if r.metadata.get("answer")]
        final = answers[0] if answers else "No relevant answer found."

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})
    st.rerun()

# =====================================================
# CHAT HISTORY
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle",
}[st.session_state.tool_mode]

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='answer'>{item['a']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
