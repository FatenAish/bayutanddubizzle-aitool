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
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =====================================================
# CSS (CENTER EVERYTHING + BUBBLES)
# =====================================================
st.markdown(
    """
    <style>
      /* Make the whole app centered + not too wide */
      section.main > div.block-container {
        max-width: 980px;
        padding-top: 2.2rem;
        padding-bottom: 2.2rem;
      }

      .center { text-align: center; }

      /* Nicer spacing between sections */
      .sp-8 { height: 8px; }
      .sp-14 { height: 14px; }
      .sp-20 { height: 20px; }

      /* Question bubbles (ONLY question) */
      .q-bubble {
        padding: 10px 14px;
        border-radius: 14px;
        width: fit-content;
        max-width: 88%;
        font-weight: 600;
        margin: 8px 0 8px;
        border: 1px solid rgba(0,0,0,0.06);
      }
      .q-general { background: #f2f2f2; }
      .q-bayut { background: #e6f4ef; }      /* light green */
      .q-dubizzle { background: #fdeaea; }   /* light red */

      .answer {
        margin: 0 0 14px 6px;
        line-height: 1.6;
      }

      /* Tighten segmented controls row */
      div[data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: center;
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
    """
    Make output clean + avoid dumping Q/A blocks.
    """
    if not text:
        return ""
    t = text.strip()

    # If it contains an A: block, take it (first one)
    m = re.search(r"\bA\s*[:–-]\s*(.*?)(?=\n\s*Q\d*\s*[:–-]|\Z)", t, flags=re.I | re.S)
    if m:
        t = m.group(1).strip()
    else:
        # Remove leading Q/A markers
        t = re.sub(r"^\s*Q\d*\s*[:–-]\s*", "", t, flags=re.I)
        t = re.sub(r"^\s*A\s*[:–-]\s*", "", t, flags=re.I)
        # Cut if another Q appears
        t = re.split(r"\n\s*Q\d*\s*[:–-]\s*", t, maxsplit=1, flags=re.I)[0]

    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\bA\s*[:–-]\s*", "", t, flags=re.I).strip()
    return t

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================================
# BUILD INDEXES
# =====================================================
@st.cache_resource
def build_indexes():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

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

VS_GENERAL, VS_BAYUT, VS_DUBIZZLE = build_indexes()

def get_vs(mode: str):
    if mode == "Bayut" and VS_BAYUT:
        return VS_BAYUT
    if mode == "Dubizzle" and VS_DUBIZZLE:
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
    <div class="center" style="color:#666;margin-bottom:10px;">Internal AI Assistant</div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='sp-14'></div>", unsafe_allow_html=True)

# =====================================================
# TOOL MODE (CLEAN + CENTERED)
# =====================================================
tool = st.segmented_control(
    "",
    options=["General", "Bayut", "Dubizzle"],
    default=st.session_state.tool_mode,
    key="tool_segmented"
)
st.session_state.tool_mode = tool

st.markdown(
    f"<h3 class='center' style='margin-top:14px;margin-bottom:6px;'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE (CLEAN + CENTERED)
# =====================================================
answer_mode = st.segmented_control(
    "",
    options=["Ultra-Fast", "Thinking"],
    default=st.session_state.answer_mode,
    key="answer_segmented"
)
st.session_state.answer_mode = answer_mode

st.markdown("<div class='sp-20'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT CARD (CENTERED LOOK)
# =====================================================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")

    b1, b2, b3 = st.columns([1, 1, 1])
    with b2:
        col_ask, col_clear = st.columns([1, 1])
        ask = col_ask.form_submit_button("Ask", use_container_width=True)
        clear = col_clear.form_submit_button("Clear chat", use_container_width=True)

# =====================================================
# CLEAR CHAT
# =====================================================
if proven_clear := clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWERING
# =====================================================
if ask and q:
    thinking = (st.session_state.answer_mode == "Thinking")
    vs = get_vs(st.session_state.tool_mode)

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.6)

    k = 4 if thinking else 1
    results = vs.similarity_search(q, k=k)

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

        if not parts:
            answer = "No relevant information found in internal files."
        else:
            # Ultra-Fast: short (first)
            # Thinking: combine (more detail)
            answer = parts[0] if not thinking else "\n\n".join(parts)

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": answer})
    st.rerun()

# =====================================================
# CHAT HISTORY (QUESTION ONLY IN BUBBLE)
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle"
}[st.session_state.tool_mode]

history = st.session_state.chat[st.session_state.tool_mode]
for item in reversed(history):
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='answer'>{item['a']}</div>", unsafe_allow_html=True)
    st.markdown("---")
