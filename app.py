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
# UI STYLES (STABLE / NO BACKGROUND)
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

      /* ----- OPTIONAL: hide Streamlit auto header spacer/odd small text (safe) ----- */
      header { background: transparent !important; }
      /* Some Streamlit builds render a tiny app label near the top; hide empty caption containers */
      div[data-testid="stCaptionContainer"] { display: none !important; }

      /* QUESTION BUBBLES */
      .q-bubble{
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 85%;
        width: fit-content;
        font-weight: 700;
        margin: 12px 0 8px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #ffffff;
      }

      /* ANSWER BUBBLES (YOU ASKED: General gray / Bayut green / dubizzle red) */
      .a-bubble{
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 92%;
        width: fit-content;
        margin: 6px 0 18px 6px;
        line-height: 1.7;
        border: 1px solid rgba(0,0,0,0.06);
        white-space: pre-wrap;
      }
      .a-general { background:#f2f2f2; }
      .a-bayut { background:#e6f4ef; border-color: rgba(14,138,109,0.22); }
      .a-dubizzle { background:#fdeaea; border-color: rgba(215,25,32,0.22); }

      /* BUTTONS */
      div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
      }

      /* Custom label (since we collapse Streamlit label) */
      .ask-label{
        font-weight: 800;
        margin: 8px 0 6px;
        font-size: 1rem;
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
    if "bayut" in n:
        return "Bayut"
    if "dubizzle" in n:
        return "Dubizzle"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text: str):
    pattern = re.compile(
        r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)",
        re.S | re.I
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = [primary] + extras
    cleaned = []
    for x in out:
        if x and x not in cleaned:
            cleaned.append(x)
    return "\n\n".join(cleaned[:4])

# =====================================================
# EMBEDDINGS
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
# HEADER (CENTERED + BRAND COLORS)
# =====================================================
st.markdown(
    """
    <div class="center">
      <h1 style="margin-bottom:4px;">
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">dubizzle</span>
        AI Content Assistant
      </h1>
      <div style="font-size:1.05rem; opacity:0.75;">
        Your Internal AI Assistant
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TOOL MODE BUTTONS
# =====================================================
tool_cols = st.columns([2, 3, 3, 3, 2])
with tool_cols[1]:
    if st.button("General", use_container_width=True):
        st.session_state.tool_mode = "General"
with tool_cols[2]:
    if st.button("Bayut", use_container_width=True):
        st.session_state.tool_mode = "Bayut"
with tool_cols[3]:
    if st.button("dubizzle", use_container_width=True):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center' style='margin-top:14px;'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE BUTTONS (Ultra-Fast / Thinking)
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True):
        st.session_state.answer_mode = "Thinking"

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT (FORCED LABEL: Ask me Anything in bold)
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)
q = st.text_input("", label_visibility="collapsed", key="q_input")

btn_cols = st.columns([1, 1])
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

        final = (
            answers[0]
            if not thinking
            else format_thinking_answer(answers[0], answers[1:])
        ) if answers else "No relevant answer found."

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})
    st.rerun()

# =====================================================
# CHAT HISTORY
# =====================================================
answer_class = {
    "General": "a-general",
    "Bayut": "a-bayut",
    "Dubizzle": "a-dubizzle",
}[st.session_state.tool_mode]

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(
        f"<div class='q-bubble'>{html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='a-bubble {answer_class}'>{html.escape(item['a'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
