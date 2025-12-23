import os
import re
import html
import time
import hashlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

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
# CSS (KEEP SIMPLE + YOUR BUBBLES)
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

      div.stButton > button{ border-radius: 10px; }
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

def is_sop_file(filename: str) -> bool:
    # Treat ANY file containing "SOP" as download-only
    return "sop" in filename.lower()

def bucket_from_filename(filename: str) -> str:
    n = filename.lower()
    if "both" in n:
        return "both"
    # MyBayut is Bayut bucket
    if "mybayut" in n or "bayut" in n:
        return "bayut"
    if "dubizzle" in n:
        return "dubizzle"
    return "general"

def parse_qa_pairs(text: str):
    """
    Extract Q/A pairs:
      Q: ...
      A: ...
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

def normalize_download_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())

def find_sop_matches(query: str):
    """
    If user asks to download SOP, show matching SOP files.
    Examples:
      download newsletter sop
      download algolia sop
      download sop
    """
    t = normalize_download_query(query)
    if ("download" not in t) and ("sop" not in t) and ("file" not in t):
        return []

    # List SOP files that actually exist in /data
    sop_files = []
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if os.path.isfile(os.path.join(DATA_DIR, f)) and is_sop_file(f):
                sop_files.append(f)

    if not sop_files:
        return []

    # If they said "download sop" only -> show all
    if "download" in t and "sop" in t and len(t.split()) <= 3:
        return sorted(sop_files)

    # Otherwise filter by keywords
    keywords = [w for w in re.split(r"[^a-z0-9]+", t) if w and w not in {"download", "sop", "file", "the", "a", "an"}]
    if not keywords:
        return sorted(sop_files)

    matches = []
    for f in sop_files:
        fn = f.lower()
        if any(k in fn for k in keywords):
            matches.append(f)

    # fallback: if no match, show all SOPs
    return sorted(matches) if matches else sorted(sop_files)

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    """
    "Thinking" should feel more detailed without an LLM:
    - show best answer
    - add extra relevant details if different
    """
    out = []
    if primary:
        out.append(primary.strip())

    for ex in extras:
        ex = ex.strip()
        if not ex:
            continue
        # skip duplicates
        if primary and ex.lower() == primary.lower():
            continue
        out.append(ex)

    # limit
    out = out[:4]
    return "\n\n".join(out) if out else "No relevant information found in internal files."

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD VECTORS (SMART Q/A INDEXING)
# We embed ONLY the question, store answer in metadata.
# =====================================================
@st.cache_resource
def build_stores():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    emb = get_embeddings()

    docs_general_all = []   # General mode: all QAs
    docs_bayut = []         # Bayut mode: Bayut/MyBayut + Both QAs only
    docs_dubizzle = []      # Dubizzle mode: Dubizzle + Both QAs only

    # Also collect SOP files list
    sop_files_existing = []

    for fname in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fp):
            continue
        if fname.startswith("."):
            continue

        if is_sop_file(fname):
            sop_files_existing.append(fname)
            continue  # SOPs NOT indexed for Q&A answers

        text = read_text(fp)
        pairs = parse_qa_pairs(text)

        # Only index Q/A pairs (that’s what you want)
        if not pairs:
            continue

        bucket = bucket_from_filename(fname)

        for q, a in pairs:
            # embed question ONLY (this is the logic fix)
            doc = Document(
                page_content=q,
                metadata={"answer": a, "source": fname, "bucket": bucket}
            )
            docs_general_all.append(doc)

            if bucket in ("bayut", "both"):
                docs_bayut.append(doc)
            if bucket in ("dubizzle", "both"):
                docs_dubizzle.append(doc)

            if bucket == "general":
                # general QAs should be available everywhere only in General mode (your rule)
                pass

    if not docs_general_all:
        raise RuntimeError("❌ No Q&A pairs found. Make sure your files contain lines starting with 'Q:' and 'A:'.")

    vs_general_all = FAISS.from_documents(docs_general_all, emb)
    vs_bayut = FAISS.from_documents(docs_bayut, emb) if docs_bayut else None
    vs_dubizzle = FAISS.from_documents(docs_dubizzle, emb) if docs_dubizzle else None

    return vs_general_all, vs_bayut, vs_dubizzle, sorted(sop_files_existing)

try:
    VS_ALL, VS_BAYUT, VS_DUBIZZLE, SOP_FILES_EXISTING = build_stores()
except Exception as e:
    st.error(str(e))
    st.stop()

def pick_store(mode: str):
    # Your rule:
    # Bayut => Bayut QA only
    # Dubizzle => Dubizzle QA only
    # General => all
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
# ANSWER MODE BUTTONS (SMALLER)
# =====================================================
mode_cols = st.columns([4, 2, 2, 4])
with mode_cols[1]:
    if st.button("Ultra-Fast", key="btn_mode_fast"):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", key="btn_mode_thinking"):
        st.session_state.answer_mode = "Thinking"

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT + ASK/CLEAR
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
    # 1) SOP download request (inside chat)
    sop_matches = find_sop_matches(q)
    if sop_matches:
        st.session_state.chat[st.session_state.tool_mode].append(
            {"type": "download", "q": q, "files": sop_matches}
        )
        st.rerun()

    # 2) Q&A answer
    thinking = (st.session_state.answer_mode == "Thinking")
    vs = pick_store(st.session_state.tool_mode)

    # If Bayut/Dubizzle store is empty, be explicit
    if st.session_state.tool_mode == "Bayut" and vs is None:
        answer = "No Bayut/MyBayut Q&A files were detected for indexing."
        st.session_state.chat[st.session_state.tool_mode].append({"type": "qa", "q": q, "a": answer})
        st.rerun()

    if st.session_state.tool_mode == "Dubizzle" and vs is None:
        answer = "No dubizzle Q&A files were detected for indexing."
        st.session_state.chat[st.session_state.tool_mode].append({"type": "qa", "q": q, "a": answer})
        st.rerun()

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.4)

    # wider search, then we pick answers from metadata
    k = 8 if thinking else 4
    results = vs.similarity_search(q, k=k)

    answers = []
    for r in results:
        a = (r.metadata.get("answer") or "").strip()
        if not a:
            continue
        # avoid title-only junk
        if len(a) < 25 and not re.search(r"[.!?،:;-]", a):
            continue
        # de-dup
        if a.lower() in [x.lower() for x in answers]:
            continue
        answers.append(a)

    if not answers:
        if st.session_state.tool_mode == "Bayut":
            final = "No relevant answer found in Bayut/MyBayut Q&A."
        elif st.session_state.tool_mode == "Dubizzle":
            final = "No relevant answer found in dubizzle Q&A."
        else:
            final = "No relevant answer found in internal Q&A."
    else:
        if not thinking:
            final = answers[0]
        else:
            final = format_thinking_answer(answers[0], answers[1:])

    st.session_state.chat[st.session_state.tool_mode].append({"type": "qa", "q": q, "a": final})
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
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item.get('q',''))}</div>",
        unsafe_allow_html=True
    )

    if item.get("type") == "download":
        files = item.get("files", [])
        if not files:
            st.markdown("<div class='answer'>No SOP files matched your request.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='answer'><b>Download SOP file(s):</b></div>", unsafe_allow_html=True)
            for f in files:
                fp = os.path.join(DATA_DIR, f)
                if not os.path.isfile(fp):
                    continue
                with open(fp, "rb") as bf:
                    st.download_button(
                        label=f"Download: {f}",
                        data=bf,
                        file_name=f,
                        mime="text/plain",
                        key=f"dl_{hashlib.md5((f+str(idx)).encode()).hexdigest()}"
                    )
    else:
        st.markdown(f"<div class='answer'>{item.get('a','')}</div>", unsafe_allow_html=True)

    st.markdown("---")
