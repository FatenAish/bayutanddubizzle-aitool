import os
import re
import html
import time
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
# UI STYLES (NO BACKGROUND IMAGE)
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

      div.stButton > button { border-radius: 10px; }

      .small-btn div.stButton > button{
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        font-size: 0.95rem !important;
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
def is_sop_file(name: str) -> bool:
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "both" in n:
        return "both"
    if "mybayut" in n or "bayut" in n:
        return "bayut"
    if "dubizzle" in n:
        return "dubizzle"
    return "general"

def read_text(fp: str) -> str:
    try:
        with open(fp, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def parse_qa_pairs(text: str):
    pairs = []
    pattern = re.compile(
        r"(?im)^\s*Q\s*[:\-–]\s*(.*?)\s*$\n^\s*A\s*[:\-–]\s*(.*?)(?=^\s*Q\s*[:\-–]\s*|\Z)",
        re.DOTALL | re.MULTILINE
    )
    for m in pattern.finditer(text):
        q = re.sub(r"\s+", " ", m.group(1)).strip()
        a = m.group(2).strip()
        a = re.sub(r"\n{3,}", "\n\n", a).strip()
        if q and a:
            pairs.append((q, a))
    return pairs

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = []
    if primary:
        out.append(primary.strip())
    for ex in extras:
        ex = ex.strip()
        if not ex:
            continue
        if primary and ex.lower() == primary.lower():
            continue
        out.append(ex)
    return "\n\n".join(out[:4]) if out else "No relevant answer found in internal Q&A."

# =====================================================
# EMBEDDINGS (✅ FIXED)
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD STORES
# =====================================================
@st.cache_resource
def build_stores():
    if not os.path.isdir(DATA_DIR):
        return None, None, None

    emb = get_embeddings()
    docs_all, docs_bayut, docs_dubizzle = [], [], []

    for fname in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fp) or fname.startswith("."):
            continue

        # skip SOPs and non-txt
        if is_sop_file(fname) or not fname.lower().endswith(".txt"):
            continue

        text = read_text(fp)
        if not text.strip():
            continue

        pairs = parse_qa_pairs(text)
        if not pairs:
            continue

        bucket = bucket_from_filename(fname)

        for q, a in pairs:
            doc = Document(page_content=q, metadata={"answer": a, "source": fname, "bucket": bucket})
            docs_all.append(doc)
            if bucket in ("bayut", "both"):
                docs_bayut.append(doc)
            if bucket in ("dubizzle", "both"):
                docs_dubizzle.append(doc)

    if not docs_all:
        return None, None, None

    vs_all = FAISS.from_documents(docs_all, emb)
    vs_bayut = FAISS.from_documents(docs_bayut, emb) if docs_bayut else None
    vs_dubizzle = FAISS.from_documents(docs_dubizzle, emb) if docs_dubizzle else None
    return vs_all, vs_bayut, vs_dubizzle

VS_ALL, VS_BAYUT, VS_DUBIZZLE = build_stores()

def pick_store(mode: str):
    if mode == "Bayut":
        return VS_BAYUT
    if mode == "Dubizzle":
        return VS_DUBIZZLE
    return VS_ALL

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
# TOOL MODE BUTTONS (CENTERED)
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
# ANSWER MODE BUTTONS (CENTERED) + SMALLER
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("Ultra-Fast", use_container_width=True, key="btn_mode_fast"):
        st.session_state.answer_mode = "Ultra-Fast"
    st.markdown("</div>", unsafe_allow_html=True)

with mode_cols[2]:
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("Thinking", use_container_width=True, key="btn_mode_thinking"):
        st.session_state.answer_mode = "Thinking"
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT + ASK/CLEAR (UNCHANGED)
# =====================================================
outer = st.columns([1, 6, 1])
with outer[1]:
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")
        bcols = st.columns([1, 1])
        ask = bcols[0].form_submit_button("Ask", use_container_width=True)
        clear = bcols[1].form_submit_button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWER
# =====================================================
if ask and q:
    thinking = (st.session_state.answer_mode == "Thinking")
    vs = pick_store(st.session_state.tool_mode)

    if vs is None:
        st.session_state.chat[st.session_state.tool_mode].append(
            {"type": "qa", "q": q, "a": "No Q&A data detected in /data."}
        )
        st.rerun()

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.35)

    k = 8 if thinking else 4
    results = vs.similarity_search(q, k=k)

    answers = []
    for r in results:
        a = (r.metadata.get("answer") or "").strip()
        if not a:
            continue
        if a.lower() in [x.lower() for x in answers]:
            continue
        answers.append(a)

    if not answers:
        final = "No relevant answer found in internal Q&A."
    else:
        final = answers[0] if not thinking else format_thinking_answer(answers[0], answers[1:])

    st.session_state.chat[st.session_state.tool_mode].append({"type": "qa", "q": q, "a": final})
    st.rerun()

# =====================================================
# CHAT HISTORY (ONLY CURRENT MODE)
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle",
}[st.session_state.tool_mode]

history = st.session_state.chat.get(st.session_state.tool_mode, [])

for item in reversed(history):
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item.get('q',''))}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='answer'>{item.get('a','')}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
