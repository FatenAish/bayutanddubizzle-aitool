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
# FORCE UI CHANGE (CSS)
# =====================================================
st.markdown(
    """
    <style>
      .center { text-align: center; }

      .mode-row {
        display: flex;
        justify-content: center;
        gap: 16px;
        margin: 10px 0 20px;
      }

      .bubble {
        padding: 10px 14px;
        border-radius: 14px;
        margin-bottom: 6px;
        max-width: 80%;
        font-weight: 500;
      }

      .bubble-general { background: #f2f2f2; }
      .bubble-bayut { background: #e6f4ef; }
      .bubble-dubizzle { background: #fdeaea; }

      .answer {
        margin-left: 6px;
        margin-bottom: 14px;
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
def read_text(fp):
    try:
        return open(fp, encoding="utf-8").read()
    except UnicodeDecodeError:
        return open(fp, encoding="utf-8-sig").read()

def clean_answer(t):
    t = re.sub(r"\bQ\s*[:–-].*", "", t, flags=re.I)
    t = re.sub(r"\bA\s*[:–-]\s*", "", t, flags=re.I)
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
# BUILD INDEX
# =====================================================
@st.cache_resource
def build_indexes():
    splitter = RecursiveCharacterTextSplitter(900, 150)
    g, b, d = [], [], []

    for f in os.listdir(DATA_DIR):
        if not f.endswith(".txt"):
            continue

        text = read_text(os.path.join(DATA_DIR, f))
        chunks = splitter.split_text(text)

        for c in chunks:
            doc = Document(page_content=c, metadata={"src": f})
            g.append(doc)
            if "bayut" in f.lower():
                b.append(doc)
            elif "dubizzle" in f.lower():
                d.append(doc)

    emb = get_embeddings()
    return (
        FAISS.from_documents(g, emb),
        FAISS.from_documents(b, emb) if b else None,
        FAISS.from_documents(d, emb) if d else None
    )

VS_G, VS_B, VS_D = build_indexes()

def vs():
    if st.session_state.tool_mode == "Bayut" and VS_B:
        return VS_B
    if st.session_state.tool_mode == "Dubizzle" and VS_D:
        return VS_D
    return VS_G

# =====================================================
# TITLE
# =====================================================
st.markdown(
    """
    <h1 class="center">
      <span style="color:#0E8A6D">Bayut</span> &
      <span style="color:#D71920">Dubizzle</span>
      AI Content Assistant
    </h1>
    <p class="center">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MODE SELECT
# =====================================================
m1, m2, m3 = st.columns(3)
with m1:
    if st.button("General", key="mode-general"):
        st.session_state.tool_mode = "General"
with m2:
    if st.button("Bayut", key="mode-bayut"):
        st.session_state.tool_mode = "Bayut"
with m3:
    if st.button("Dubizzle", key="mode-dubizzle"):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE (REAL TOGGLE)
# =====================================================
st.markdown("<div class='mode-row'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("Ultra-Fast", key="fast-btn"):
        st.session_state.answer_mode = "Ultra-Fast"
with c2:
    if st.button("Thinking", key="think-btn"):
        st.session_state.answer_mode = "Thinking"

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# QUESTION INPUT
# =====================================================
with st.form("ask-form", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here…")
    ask = st.form_submit_button("Ask")

# =====================================================
# CLEAR CHAT
# =====================================================
if st.button("Clear chat", key="clear-chat"):
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWER
# =====================================================
if ask and q:
    k = 4 if st.session_state.answer_mode == "Thinking" else 1

    if st.session_state.answer_mode == "Thinking":
        with st.spinner("Thinking…"):
            time.sleep(0.6)

    results = vs().similarity_search(q, k=k)
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
    "General": "bubble-general",
    "Bayut": "bubble-bayut",
    "Dubizzle": "bubble-dubizzle"
}[st.session_state.tool_mode]

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(
        f"<div class='bubble {bubble_class}'>{html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='answer'>{item['a']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
