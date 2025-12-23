import os
import re
import html
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# =====================================================
# CSS (CENTER LAYOUT + BUBBLES + SMALLER MODE BUTTONS)
# =====================================================
st.markdown(
    """
    <style>
      /* Keep content centered and not super wide */
      section.main > div.block-container{
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
      }

      .center { text-align:center; }

      .q-bubble{
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        width: fit-content;
        font-weight: 600;
        margin: 10px 0 8px;
        border: 1px solid rgba(0,0,0,0.06);
      }
      .q-general{ background:#f2f2f2; }
      .q-bayut{ background:#e6f4ef; }     /* light green */
      .q-dubizzle{ background:#fdeaea; }  /* light red */

      .answer{
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
      }

      /* General button look */
      div.stButton > button{
        border-radius: 10px;
      }

      /* ✅ ONLY shrink Ultra-Fast + Thinking buttons (by key) */
      div[data-testid="stButton"][data-testid*="btn_mode_fast"] > button,
      div[data-testid="stButton"][data-testid*="btn_mode_thinking"] > button {
        padding: 0.35rem 0.65rem !important;
        font-size: 0.92rem !important;
        min-height: 38px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("answer_mode", "Ultra-Fast")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# =====================================================
# HELPERS
# =====================================================
def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def clean_answer(text: str) -> str:
    """Clean to a single readable answer (prevents Q/A dumps)."""
    if not text:
        return ""
    t = text.strip()

    m = re.search(r"\bA\s*[:–-]\s*(.*?)(?=\n\s*Q\d*\s*[:–-]|\Z)", t, flags=re.I | re.S)
    if m:
        t = m.group(1).strip()
    else:
        t = re.sub(r"^\s*Q\d*\s*[:–-]\s*", "", t, flags=re.I)
        t = re.sub(r"^\s*A\s*[:–-]\s*", "", t, flags=re.I)
        t = re.split(r"\n\s*Q\d*\s*[:–-]\s*", t, maxsplit=1, flags=re.I)[0]

    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\bA\s*[:–-]\s*", "", t, flags=re.I).strip()
    return t

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD INDEXES
# =====================================================
@st.cache_resource
def build_indexes():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    general_docs, bayut_docs, dubizzle_docs = [], [], []

    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        raw = read_text(os.path.join(DATA_DIR, fname))
        chunks = splitter.split_text(raw)

        lname = fname.lower()
        bucket = "general"
        if "bayut" in lname:
            bucket = "bayut"
        elif "dubizzle" in lname:
            bucket = "dubizzle"

        for c in chunks:
            doc = Document(page_content=c, metadata={"source": fname})
            general_docs.append(doc)
            if bucket == "bayut":
                bayut_docs.append(doc)
            elif bucket == "dubizzle":
                dubizzle_docs.append(doc)

    if not general_docs:
        raise RuntimeError("❌ No readable .txt files found in /data")

    emb = get_embeddings()
    vs_general = FAISS.from_documents(general_docs, emb)
    vs_bayut = FAISS.from_documents(bayut_docs, emb) if bayut_docs else None
    vs_dubizzle = FAISS.from_documents(dubizzle_docs, emb) if dubizzle_docs else None
    return vs_general, vs_bayut, vs_dubizzle

try:
    VS_GENERAL, VS_BAYUT, VS_DUBIZZLE = build_indexes()
except Exception as e:
    st.error(str(e))
    st.stop()

def get_vs():
    if st.session_state.tool_mode == "Bayut" and VS_BAYUT:
        return VS_BAYUT
    if st.session_state.tool_mode == "Dubizzle" and VS_DUBIZZLE:
        return VS_DUBIZZLE
    return VS_GENERAL

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 class="center" style="font-weight:900;margin-bottom:6px;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span> AI Content Assistant
    </h1>
    <div class="center" style="color:#666;margin-bottom:14px;">Internal AI Assistant</div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TOOL MODE (CENTERED SEPARATE BUTTONS)
# =====================================================
tool_cols = st.columns([2, 3, 3, 3, 2])
with tool_cols[1]:
    if st.button("General", use_container_width=True, key="btn_tool_general"):
        st.session_state.tool_mode = "General"
with tool_cols[2]:
    if st.button("Bayut", use_container_width=True, key="btn_tool_bayut"):
        st.session_state.tool_mode = "Bayut"
with tool_cols[3]:
    if st.button("Dubizzle", use_container_width=True, key="btn_tool_dubizzle"):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center' style='margin-top:18px;margin-bottom:6px;'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE (CENTERED SEPARATE BUTTONS) ✅ SMALLER
# =====================================================
mode_cols = st.columns([3, 4, 4, 3])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True, key="btn_mode_fast"):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True, key="btn_mode_thinking"):
        st.session_state.answer_mode = "Thinking"

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT (CENTERED) + ASK/CLEAR SAME ROW
# =====================================================
outer = st.columns([1, 6, 1])
with outer[1]:
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")

        bcols = st.columns([1, 1])
        ask = bcols[0].form_submit_button("Ask", use_container_width=True)
        clear = bcols[1].form_submit_button("Clear chat", use_container_width=True)

# =====================================================
# CLEAR CHAT
# =====================================================
if clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWERING
# =====================================================
if ask and q:
    thinking = (st.session_state.answer_mode == "Thinking")
    k = 4 if thinking else 1

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.6)

    results = get_vs().similarity_search(q, k=k)

    if not results:
        answer = "No relevant information found in internal files."
    else:
        parts, seen = [], set()
        for r in results:
            a = clean_answer(r.page_content)
            if not a:
                continue
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(a)

        answer = parts[0] if not thinking else "\n\n".join(parts)
        if not answer:
            answer = "No relevant information found in internal files."

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": answer})
    st.rerun()

# =====================================================
# CHAT HISTORY (QUESTION ONLY IN BUBBLE)
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
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
