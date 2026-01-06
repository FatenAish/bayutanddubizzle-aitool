import os
import re
import html
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# =====================================================
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================
# HARD-FORCE STYLES (VERY SPECIFIC + !important)
# =====================================================
st.markdown(
    """
    <style>
      /* Force container width */
      section.main > div.block-container{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
      }

      /* Hide Streamlit caption containers (the "Internal AI Assistant" you keep seeing) */
      div[data-testid="stCaptionContainer"]{ display:none !important; }
      div[data-testid="stHeader"]{ background: transparent !important; }
      header{ background: transparent !important; }

      /* Center ALL h1 inside the app */
      section.main h1{
        text-align:center !important;
        margin-bottom: 6px !important;
      }

      /* Brand header block */
      #brand-header{
        text-align:center !important;
        margin: 0 0 10px 0 !important;
      }
      #brand-title{
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        line-height: 1.15 !important;
      }
      #brand-subtitle{
        margin-top: 6px !important;
        font-size: 1.05rem !important;
        opacity: 0.75 !important;
      }

      /* Buttons */
      div.stButton > button{
        border-radius: 12px !important;
        font-weight: 600 !important;
      }

      /* Custom label for input */
      .ask-label{
        font-weight: 800 !important;
        font-size: 1rem !important;
        margin: 12px 0 6px 0 !important;
      }

      /* QUESTION bubble (same for all modes) */
      .q-bubble{
        padding: 12px 16px !important;
        border-radius: 16px !important;
        max-width: 85% !important;
        width: fit-content !important;
        font-weight: 700 !important;
        margin: 12px 0 8px 0 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        background: #ffffff !important;
      }

      /* ANSWER bubbles per mode */
      .a-bubble{
        padding: 12px 16px !important;
        border-radius: 16px !important;
        max-width: 92% !important;
        width: fit-content !important;
        margin: 6px 0 18px 6px !important;
        line-height: 1.7 !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        white-space: normal !important;
      }
      .a-general{ background:#f2f2f2 !important; }
      .a-bayut{ background:#e6f4ef !important; border-color: rgba(14,138,109,0.22) !important; }
      .a-dubizzle{ background:#fdeaea !important; border-color: rgba(215,25,32,0.22) !important; }

      /* Center the "Assistant" heading */
      .mode-title{
        text-align:center !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 18px 0 10px 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
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
    return re.sub(r"\s+", " ", (text or "").lower().strip())

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
    pattern = re.compile(r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)", re.S | re.I)
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = [primary] + extras
    cleaned = []
    for x in out:
        x = (x or "").strip()
        if x and x not in cleaned:
            cleaned.append(x)
    return "\n\n".join(cleaned[:4])

def br(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")

# =====================================================
# SMART Q&A LOGIC HELPERS
# =====================================================
GREETINGS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "السلام عليكم"
}

def is_greeting(q: str) -> bool:
    return normalize(q) in GREETINGS

def is_entity_question(q: str) -> bool:
    qn = normalize(q)
    return qn.startswith(("who is", "what is", "who are"))

def extract_entity(q: str) -> str:
    qn = normalize(q)
    qn = re.sub(r"^(who is|what is|who are)\s+", "", qn).strip()
    qn = re.sub(r"[?!.]+$", "", qn).strip()
    return qn

def safe_merge_facts(facts: list[str]) -> str:
    uniq = []
    for f in facts:
        f = (f or "").strip()
        if f and f not in uniq:
            uniq.append(f)
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    return " ".join(uniq)

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# DATA FINGERPRINT (AUTO-REFRESH WHEN FILES CHANGE)
# =====================================================
def data_fingerprint():
    """
    Makes Streamlit cache rebuild ONLY when data files change
    (so you don't need to "reboot" to pick up updates).
    """
    items = []
    if not os.path.isdir(DATA_DIR):
        return tuple()
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        try:
            stt = os.stat(fp)
            items.append((f, int(stt.st_mtime), int(stt.st_size)))
        except Exception:
            continue
    return tuple(sorted(items))

# =====================================================
# BUILD STORES (IMPROVED: EMBED Q + A, PLUS EXACT Q MAP)
# =====================================================
@st.cache_resource
def build_stores(_fingerprint):
    """
    IMPORTANT CHANGE:
    - We embed BOTH question and answer into page_content for better retrieval.
    - We also create an exact normalized question->answer map to prevent wrong matches.
    """
    if not os.path.isdir(DATA_DIR):
        return None

    emb = get_embeddings()

    docs = {"General": [], "Bayut": [], "Dubizzle": []}
    qmap = {"General": {}, "Bayut": {}, "Dubizzle": {}}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        b = bucket_from_filename(f)

        for q, a in parse_qa_pairs(read_text(fp)):
            nq = normalize(q)
            if nq:
                # exact lookup maps
                qmap["General"][nq] = a
                qmap[b][nq] = a

            # embed Q + A (this is the biggest accuracy fix)
            content = f"Q: {q}\nA: {a}"
            doc = Document(
                page_content=content,
                metadata={"question": q, "answer": a, "bucket": b, "file": f}
            )

            docs["General"].append(doc)
            docs[b].append(doc)

    vs = {
        "General": FAISS.from_documents(docs["General"], emb) if docs["General"] else None,
        "Bayut": FAISS.from_documents(docs["Bayut"], emb) if docs["Bayut"] else None,
        "Dubizzle": FAISS.from_documents(docs["Dubizzle"], emb) if docs["Dubizzle"] else None,
    }

    return {"vs": vs, "docs": docs, "qmap": qmap}

STORE = build_stores(data_fingerprint())

def pick_assets():
    mode = st.session_state.tool_mode
    if not STORE:
        return None, [], {}
    return STORE["vs"].get(mode), STORE["docs"].get(mode, []), STORE["qmap"].get(mode, {})

# =====================================================
# HEADER (FORCED CENTER + BRAND COLORS)
# =====================================================
st.markdown(
    """
    <div id="brand-header">
      <div id="brand-title">
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style
