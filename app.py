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
# STYLES
# =====================================================
st.markdown(
    """
    <style>
      section.main > div.block-container{
        max-width: 980px !important;
        padding: 2rem !important;
      }

      .ask-label{ font-weight:800; margin:12px 0 6px; }

      .q-bubble{
        padding:12px 16px;
        border-radius:16px;
        font-weight:700;
        margin:12px 0 8px;
        background:#fff;
        border:1px solid rgba(0,0,0,0.08);
        max-width:85%;
      }

      .a-bubble{
        padding:12px 16px;
        border-radius:16px;
        margin:6px 0 18px 6px;
        line-height:1.7;
        border:1px solid rgba(0,0,0,0.06);
        max-width:92%;
      }
      .a-general{ background:#f2f2f2; }
      .a-bayut{ background:#e6f4ef; }
      .a-dubizzle{ background:#fdeaea; }
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
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

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

def br(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")

# =====================================================
# GREETINGS + ENTITY LOGIC
# =====================================================
GREETINGS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "السلام عليكم"
}

def is_greeting(q: str) -> bool:
    return normalize(q) in GREETINGS

def is_entity_question(q: str) -> bool:
    return normalize(q).startswith(("who is", "what is", "who are"))

def entity_lookup_aggregate(q: str, store):
    """
    Aggregate all facts related to the entity and merge them.
    """
    entity = normalize(q)
    entity = re.sub(r"^(who is|what is|who are)\s+", "", entity).strip()
    if not entity:
        return None

    docs = store.similarity_search(entity, k=10)

    facts = []
    for d in docs:
        ans = d.metadata.get("answer", "")
        if entity in ans.lower():
            facts.append(ans.strip())

    if not facts:
        return None

    unique = []
    for f in facts:
        if f not in unique:
            unique.append(f)

    return " ".join(unique)

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
# INPUT
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)
q = st.text_input("", label_visibility="collapsed")

ask, clear = st.columns(2)
ask = ask.button("Ask", use_container_width=True)
clear = clear.button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWER (SMART LOGIC)
# =====================================================
if ask and q:
    vs = pick_store()
    final = None

    # 1️⃣ Greeting
    if is_greeting(q):
        final = (
            "Hi! 👋 I’m the Bayut & dubizzle Internal AI Assistant. "
            "Ask your question and I’ll answer using the internal knowledge files."
        )

    # 2️⃣ Entity question
    elif vs and is_entity_question(q):
        final = entity_lookup_aggregate(q, vs)

    # 3️⃣ Fallback to semantic search
    if final is None:
        if not vs:
            final = "No internal Q&A data found."
        else:
            results = vs.similarity_search(q, k=6)
            answers = [r.metadata.get("answer") for r in results if r.metadata.get("answer")]
            final = answers[0] if answers else (
                "I couldn’t find a clear answer in the internal knowledge base."
            )

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
    st.markdown(f"<div class='q-bubble'>{br(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='a-bubble {answer_class}'>{br(item['a'])}</div>", unsafe_allow_html=True)
    st.markdown("---")
