import os
import re
import html
import time
import hashlib
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
# GLOBAL CSS (UI FIXES)
# =====================================================
st.markdown(
    """
    <style>
      /* Center mode buttons row */
      .center-row {
        display: flex;
        justify-content: center;
        gap: 18px;
        margin-top: 10px;
      }

      /* Thinking toggle buttons */
      .mode-btn {
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #ddd;
        cursor: pointer;
        font-weight: 500;
        background: #fff;
      }
      .mode-btn.active {
        border-color: #ff4b4b;
        color: #ff4b4b;
      }

      /* Question bubbles */
      .q-bubble {
        padding: 10px 14px;
        border-radius: 14px;
        margin-bottom: 6px;
        width: fit-content;
        max-width: 85%;
        font-weight: 500;
      }
      .q-general { background: #f2f2f2; }
      .q-bayut { background: #e6f4ef; }
      .q-dubizzle { background: #fdeaea; }

      /* Answer spacing */
      .answer {
        margin-left: 8px;
        margin-bottom: 16px;
      }

      /* Clear chat button */
      .clear-btn {
        margin-top: 10px;
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
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "Dubizzle": []
})

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
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\bQ\s*[:–-].*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bA\s*[:–-]\s*", "", t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()

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
        raise RuntimeError("/data folder missing")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    general, bayut, dubizzle = [], [], []

    for file in os.listdir(DATA_DIR):
        if not file.lower().endswith(".txt"):
            continue

        raw = read_text(os.path.join(DATA_DIR, file))
        chunks = splitter.split_text(raw)

        bucket = "general"
        name = file.lower()
        if "bayut" in name:
            bucket = "bayut"
        elif "dubizzle" in name:
            bucket = "dubizzle"

        for c in chunks:
            doc = Document(page_content=c, metadata={"source": file})
            general.append(doc)
            if bucket == "bayut":
                bayut.append(doc)
            elif bucket == "dubizzle":
                dubizzle.append(doc)

    emb = get_embeddings()
    return (
        FAISS.from_documents(general, emb),
        FAISS.from_documents(bayut, emb) if bayut else None,
        FAISS.from_documents(dubizzle, emb) if dubizzle else None,
    )

VS_GENERAL, VS_BAYUT, VS_DUBIZZLE = build_indexes()

def get_vs():
    if st.session_state.tool_mode == "Bayut" and VS_BAYUT:
        return VS_BAYUT
    if st.session_state.tool_mode == "Dubizzle" and VS_DUBIZZLE:
        return VS_DUBIZZLE
    return VS_GENERAL

# =====================================================
# TITLE
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    <p style="text-align:center;color:#666;">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MODE SWITCH
# =====================================================
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("General", use_container_width=True):
        st.session_state.tool_mode = "General"
with c2:
    if st.button("Bayut", use_container_width=True):
        st.session_state.tool_mode = "Bayut"
with c3:
    if st.button("Dubizzle", use_container_width=True):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 style='text-align:center'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# THINKING TOGGLE (CLEAN)
# =====================================================
t1, t2, t3 = st.columns([2, 1, 2])
with t2:
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Ultra-Fast",
            key="fast",
            use_container_width=True
        ):
            st.session_state.answer_mode = "Ultra-Fast"
    with col2:
        if st.button(
            "Thinking",
            key="think",
            use_container_width=True
        ):
            st.session_state.answer_mode = "Thinking"

# =====================================================
# QUESTION INPUT
# =====================================================
with st.form("ask", clear_on_submit=True):
    q = st.text_input(
        "",
        placeholder="Type your question here…",
        label_visibility="collapsed"
    )
    ask = st.form_submit_button("Ask")

# =====================================================
# CLEAR CHAT
# =====================================================
if st.button("Clear chat", type="secondary"):
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWERING
# =====================================================
if ask and q:
    vs = get_vs()
    k = 4 if st.session_state.answer_mode == "Thinking" else 1

    if st.session_state.answer_mode == "Thinking":
        with st.spinner("Thinking…"):
            time.sleep(0.6)

    results = vs.similarity_search(q, k=k)

    parts = []
    for r in results:
        a = clean_answer(r.page_content)
        if a and a not in parts:
            parts.append(a)

    answer = "\n\n".join(parts) if parts else "No relevant information found."

    st.session_state.chat[st.session_state.tool_mode].append({
        "q": q,
        "a": answer
    })
    st.rerun()

# =====================================================
# CHAT HISTORY (QUESTION BUBBLES ONLY)
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle"
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
