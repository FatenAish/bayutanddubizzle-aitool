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
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================
# SOP FILES (DOWNLOAD-ONLY, NOT FOR Q&A)
# =====================================================
SOP_FILES = {
    "Bayut-Algolia Locations SOP.txt",
    "Bayut-MyBayut Newsletters SOP.txt",
    "Bayut-PM Campaigns SOP.txt",
    "Bayut-Social Media Posting SOP.txt",
    "Both Corrections and Updates for Listings.txt",
    "dubizzle Newsletters SOP.txt",
    "dubizzle PM Campaigns SOP.txt",
}

SOP_KEYWORDS_MAP = [
    (["newsletter", "newsletters"], ["Bayut-MyBayut Newsletters SOP.txt", "dubizzle Newsletters SOP.txt"]),
    (["algolia", "location", "locations"], ["Bayut-Algolia Locations SOP.txt"]),
    (["pm", "performance marketing", "campaign", "campaigns"], ["Bayut-PM Campaigns SOP.txt", "dubizzle PM Campaigns SOP.txt"]),
    (["social", "instagram", "posting"], ["Bayut-Social Media Posting SOP.txt"]),
    (["correction", "corrections", "update", "updates", "listing", "listings", "project", "projects"], ["Both Corrections and Updates for Listings.txt"]),
]

# =====================================================
# CSS (CENTER + SMALL MODE BUTTONS + QUESTION BUBBLES)
# =====================================================
st.markdown(
    """
    <style>
      section.main > div.block-container{
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
      }
      .center { text-align:center; }

      /* Question bubbles (ONLY question) */
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
      .q-bayut{ background:#e6f4ef; }
      .q-dubizzle{ background:#fdeaea; }

      .answer{
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
      }

      /* Buttons look */
      div.stButton > button{
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
def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def parse_qa_pairs(text: str):
    """
    Extract Q/A pairs from text like:
      Q: ...
      A: ...
    Returns list of (q, a)
    """
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

def clean_answer(text: str) -> str:
    """Return clean answer text without Q/A markers."""
    if not text:
        return ""
    t = text.strip()
    # If full doc is 'Q: ... A: ...', keep only A content
    m = re.search(r"(?is)\bA\s*[:\-–]\s*(.*)", t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def looks_like_heading_only(s: str) -> bool:
    if not s:
        return True
    x = s.strip()
    # Very short and no punctuation => likely a heading
    if len(x) < 50 and not re.search(r"[.!?،:;-]", x):
        return True
    # SOP-ish title-only
    if re.search(r"\bSOP\b", x, flags=re.I) and len(x.split()) <= 7:
        return True
    return False

def is_bayut_qa_file(filename: str) -> bool:
    n = filename.lower()
    return ("bayut" in n) and (("qa" in n) or ("q&a" in n) or ("qanda" in n) or ("q & a" in n))

def is_dubizzle_qa_file(filename: str) -> bool:
    n = filename.lower()
    return ("dubizzle" in n) and (("qa" in n) or ("q&a" in n) or ("qanda" in n) or ("q & a" in n))

def is_sop_file(filename: str) -> bool:
    return filename in SOP_FILES

def sop_matches_query(q: str):
    t = q.lower().strip()
    if "download" not in t and "sop" not in t and "file" not in t:
        return []

    matches = set()
    for keys, files in SOP_KEYWORDS_MAP:
        if any(k in t for k in keys):
            for f in files:
                matches.add(f)

    # If user says "download sop" but no keyword, show all SOPs
    if ("download" in t and "sop" in t) and not matches:
        matches = set(SOP_FILES)

    # Only files that exist
    existing = []
    for f in matches:
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            existing.append(f)
    return existing

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD VECTORSTORES (IMPORTANT ROUTING)
# - Bayut mode: ONLY Bayut Q&A
# - Dubizzle mode: ONLY Dubizzle Q&A
# - General mode: ALL (General + Bayut Q&A + Dubizzle Q&A)
# - SOP files: excluded from Q&A, only downloadable
# =====================================================
@st.cache_resource
def build_vectorstores():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    emb = get_embeddings()

    docs_all = []
    docs_bayut_qa = []
    docs_dubizzle_qa = []

    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        # SOPs are download-only (NOT indexed)
        if is_sop_file(fname):
            continue

        full_text = read_text(os.path.join(DATA_DIR, fname))
        pairs = parse_qa_pairs(full_text)

        # If it contains Q/A, index by Q/A pairs (smart matching)
        if pairs:
            for q, a in pairs:
                content = f"Q: {q}\nA: {a}"
                doc = Document(page_content=content, metadata={"source": fname, "q": q})
                docs_all.append(doc)

                if is_bayut_qa_file(fname):
                    docs_bayut_qa.append(doc)
                if is_dubizzle_qa_file(fname):
                    docs_dubizzle_qa.append(doc)

        else:
            # Non-QA files: chunk as fallback content for GENERAL only
            chunks = splitter.split_text(full_text)
            for c in chunks:
                doc = Document(page_content=c, metadata={"source": fname})
                docs_all.append(doc)

    if not docs_all:
        raise RuntimeError("❌ No readable non-SOP .txt files found")

    vs_all = FAISS.from_documents(docs_all, emb)
    vs_bayut = FAISS.from_documents(docs_bayut_qa, emb) if docs_bayut_qa else None
    vs_dubizzle = FAISS.from_documents(docs_dubizzle_qa, emb) if docs_dubizzle_qa else None

    return vs_all, vs_bayut, vs_dubizzle

try:
    VS_ALL, VS_BAYUT_QA, VS_DUBIZZLE_QA = build_vectorstores()
except Exception as e:
    st.error(str(e))
    st.stop()

def pick_vectorstore(mode: str):
    # User requirement: Bayut uses Bayut QA only, Dubizzle uses Dubizzle QA only, General uses all.
    if mode == "Bayut":
        return VS_BAYUT_QA
    if mode == "Dubizzle":
        return VS_DUBIZZLE_QA
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
# ANSWER MODE BUTTONS (SMALLER)
# (Your screenshot: they were too big)
# =====================================================
mode_cols = st.columns([4, 2, 2, 4])  # <- smaller centered buttons
with mode_cols[1]:
    if st.button("Ultra-Fast", key="btn_mode_fast"):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", key="btn_mode_thinking"):
        st.session_state.answer_mode = "Thinking"

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT (CENTERED) + ASK/CLEAR ROW
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
# ANSWERING + SOP DOWNLOAD REQUESTS
# =====================================================
if ask and q:
    # 1) SOP downloads inside chat
    sop_files = sop_matches_query(q)
    if sop_files:
        st.session_state.chat[st.session_state.tool_mode].append({
            "type": "sop",
            "q": q,
            "sops": sop_files
        })
        st.rerun()

    # 2) Normal Q&A
    thinking = (st.session_state.answer_mode == "Thinking")
    vs = pick_vectorstore(st.session_state.tool_mode)

    # If Bayut/Dubizzle has no QA store, fail clearly
    if st.session_state.tool_mode in ["Bayut", "Dubizzle"] and vs is None:
        answer = f"No {st.session_state.tool_mode} Q&A files were found in /data."
        st.session_state.chat[st.session_state.tool_mode].append({"type": "text", "q": q, "a": answer})
        st.rerun()

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.5)

    k = 6 if thinking else 3
    results = vs.similarity_search(q, k=k)

    parts, seen = [], set()
    for r in results:
        a = clean_answer(r.page_content)
        if not a or looks_like_heading_only(a):
            continue
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(a)
        if not thinking and parts:
            break
        if thinking and len(parts) >= 4:
            break

    if not parts:
        if st.session_state.tool_mode in ["Bayut", "Dubizzle"]:
            answer = f"No relevant information found in {st.session_state.tool_mode} Q&A."
        else:
            answer = "No relevant information found in internal files."
    else:
        answer = parts[0] if not thinking else "\n\n".join(parts)

    st.session_state.chat[st.session_state.tool_mode].append({"type": "text", "q": q, "a": answer})
    st.rerun()

# =====================================================
# CHAT HISTORY (QUESTION BUBBLE ONLY)
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle",
}[st.session_state.tool_mode]

history = st.session_state.chat[st.session_state.tool_mode]

for idx, item in enumerate(reversed(history)):
    # QUESTION BUBBLE
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item.get('q',''))}</div>",
        unsafe_allow_html=True
    )

    # ANSWER / SOP DOWNLOADS
    if item.get("type") == "sop":
        files = item.get("sops", [])
        if not files:
            st.markdown("<div class='answer'>No SOP files matched your request.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='answer'><b>Download SOP file(s):</b></div>", unsafe_allow_html=True)
            for f in files:
                fp = os.path.join(DATA_DIR, f)
                try:
                    with open(fp, "rb") as bf:
                        st.download_button(
                            label=f"Download: {f}",
                            data=bf,
                            file_name=f,
                            mime="text/plain",
                            key=f"dl_{hashlib.md5((f+str(idx)).encode()).hexdigest()}"
                        )
                except Exception:
                    st.markdown(f"<div class='answer'>Could not load: {html.escape(f)}</div>", unsafe_allow_html=True)

    else:
        st.markdown(f"<div class='answer'>{item.get('a','')}</div>", unsafe_allow_html=True)

    st.markdown("---")
