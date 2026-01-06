import os
import re
import html
import time
import hashlib
from typing import Dict, List, Tuple, Optional

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
# STYLES (KEEP YOUR DESIGN)
# =====================================================
st.markdown(
    """
    <style>
      section.main > div.block-container{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
      }

      .center { text-align:center; }

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

      div.stButton > button{
        border-radius: 12px !important;
        font-weight: 600 !important;
      }

      .mode-title{
        text-align:center !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 18px 0 10px 0 !important;
      }

      .ask-label{
        font-weight: 800 !important;
        font-size: 1rem !important;
        margin: 12px 0 6px 0 !important;
      }

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
def br(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]+", " ", s)  # keep Arabic too
    toks = [t for t in s.split() if len(t) > 1]
    return toks

def is_sop_file(name: str) -> bool:
    # you said SOPs are download-only in another setup; here we index only QA files
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "dubizzle" in n:
        return "Dubizzle"
    if "bayut" in n:
        return "Bayut"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"Q[:\-\)]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-\)]|\Z)", re.S | re.I)
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def data_fingerprint(data_dir: str) -> str:
    """
    Makes cache auto-invalidate when files change,
    without needing manual reboot/refresh.
    """
    if not os.path.isdir(data_dir):
        return "no-data-dir"

    parts = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        fp = os.path.join(data_dir, fn)
        try:
            stt = os.stat(fp)
            parts.append(f"{fn}:{stt.st_size}:{int(stt.st_mtime)}")
        except Exception:
            parts.append(f"{fn}:err")
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()

def extract_subject(query: str) -> Optional[str]:
    """
    If user asks: "who is X", "what is X", return X (simple).
    Helps with name questions (Faten, Sarah, etc.)
    """
    q = normalize_text(query)
    m = re.search(r"\b(who is|what is|مين|من هو|من هي)\b\s+(.+)$", q)
    if not m:
        return None
    subj = m.group(2)
    subj = re.sub(r"[?.!]+$", "", subj).strip()
    # keep only first 3 words max (names)
    words = subj.split()
    return " ".join(words[:3]) if words else None

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD STORES (AUTO-REFRESH ON DATA CHANGE)
# =====================================================
@st.cache_resource
def build_stores(_fingerprint: str):
    # _fingerprint is only to invalidate cache automatically
    emb = get_embeddings()

    stores_docs: Dict[str, List[Document]] = {"General": [], "Bayut": [], "Dubizzle": []}

    # If /data doesn't exist, also look in repo root as a fallback
    search_dirs = [DATA_DIR]
    if not os.path.isdir(DATA_DIR):
        search_dirs = [BASE_DIR]
    else:
        # If /data exists but empty, also fallback to BASE_DIR
        if not any(fn.lower().endswith(".txt") for fn in os.listdir(DATA_DIR)):
            search_dirs.append(BASE_DIR)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue

        for f in os.listdir(d):
            if not f.lower().endswith(".txt"):
                continue
            if is_sop_file(f):
                continue

            fp = os.path.join(d, f)
            text = read_text(fp)
            pairs = parse_qa_pairs(text)

            for q, a in pairs:
                # IMPORTANT: index BOTH Q + A (fixes "who is faten" etc.)
                content = f"Q: {q}\nA: {a}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "question": q,
                        "answer": a,
                        "source_file": f,
                        "bucket": bucket_from_filename(f),
                    },
                )

                b = doc.metadata["bucket"]
                stores_docs["General"].append(doc)  # all docs searchable in General
                if b in ("Bayut", "Dubizzle"):
                    stores_docs[b].append(doc)

    vs_all = FAISS.from_documents(stores_docs["General"], emb) if stores_docs["General"] else None
    vs_bayut = FAISS.from_documents(stores_docs["Bayut"], emb) if stores_docs["Bayut"] else None
    vs_dubizzle = FAISS.from_documents(stores_docs["Dubizzle"], emb) if stores_docs["Dubizzle"] else None

    # Also return docs so we can do keyword fallback (fast + smart)
    return (vs_all, vs_bayut, vs_dubizzle, stores_docs)

FP = data_fingerprint(DATA_DIR)
VS_ALL, VS_BAYUT, VS_DUBIZZLE, DOCS = build_stores(FP)

def pick_store():
    return {
        "General": VS_ALL,
        "Bayut": VS_BAYUT,
        "Dubizzle": VS_DUBIZZLE,
    }[st.session_state.tool_mode]

def pick_docs():
    return {
        "General": DOCS.get("General", []),
        "Bayut": DOCS.get("Bayut", []),
        "Dubizzle": DOCS.get("Dubizzle", []),
    }[st.session_state.tool_mode]

# =====================================================
# SMART RETRIEVAL (CHATGPT-STYLE FROM YOUR FILES)
# =====================================================
def smart_answer(query: str, k_vec: int, thinking: bool) -> str:
    vs = pick_store()
    docs = pick_docs()

    if vs is None or not docs:
        return "No internal Q&A data found for this mode."

    subject = extract_subject(query)
    q_tokens = set(tokenize(query))

    # 1) Vector candidates (with scores)
    try:
        vec_hits = vs.similarity_search_with_score(query, k=k_vec)
    except Exception:
        # fallback if score method not available
        vec_docs = vs.similarity_search(query, k=k_vec)
        vec_hits = [(d, 1.0) for d in vec_docs]

    # Convert vector distance -> similarity
    candidates = []
    for d, dist in vec_hits:
        text = normalize_text(d.page_content)
        # smaller dist = better; convert safely
        sim = 1.0 / (1.0 + float(dist))
        candidates.append((d, sim))

    # 2) Keyword/name fallback boost (super important for "who is faten")
    subj_norm = normalize_text(subject) if subject else None

    def lexical_score(d: Document) -> float:
        hay = normalize_text(d.page_content)
        toks = set(tokenize(hay))
        overlap = len(q_tokens & toks) / max(1, len(q_tokens))

        subj_bonus = 0.0
        if subj_norm and subj_norm in hay:
            subj_bonus = 1.2  # strong boost if the subject appears in Q/A text

        # also boost if any query token appears in the ANSWER specifically
        ans = normalize_text(d.metadata.get("answer", ""))
        ans_bonus = 0.0
        for t in list(q_tokens)[:8]:
            if t in ans:
                ans_bonus += 0.15

        return overlap + subj_bonus + ans_bonus

    scored = []
    for d, sim in candidates:
        score = (1.2 * sim) + (0.9 * lexical_score(d))
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d in scored[: (4 if thinking else 2)]]

    # If subject exists but vector missed, do a pure keyword search in docs
    if subj_norm:
        kw_matches = []
        for d in docs:
            hay = normalize_text(d.page_content)
            if subj_norm in hay:
                kw_matches.append(d)
        # Merge keyword matches at front
        for d in kw_matches[:3]:
            if d not in top_docs:
                top_docs.insert(0, d)

    # Build final answer by merging unique answers (no AI rewrite, just smart merge)
    answers = []
    for d in top_docs:
        a = (d.metadata.get("answer") or "").strip()
        if a and a not in answers:
            answers.append(a)

    if not answers:
        return "No relevant answer found."

    if not thinking:
        return answers[0]

    # Thinking: show up to 3 unique blocks
    return "\n\n".join(answers[:3])

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div id="brand-header">
      <div id="brand-title">
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">dubizzle</span>
        AI Content Assistant
      </div>
      <div id="brand-subtitle">Your Internal AI Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
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

st.markdown(f"<div class='mode-title'>{st.session_state.tool_mode} Assistant</div>", unsafe_allow_html=True)

# =====================================================
# ANSWER MODE BUTTONS
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True):
        st.session_state.answer_mode = "Thinking"

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT
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
    thinking = st.session_state.answer_mode == "Thinking"
    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.15)

    final = smart_answer(q, k_vec=(12 if thinking else 6), thinking=thinking)

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
