import os
import re
import html
import time
import shutil
import hashlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# 🔐 HARD ACCESS GATE (SESSION + URL LOCK) — MUST BE FIRST
# =====================================================
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
REQUIRE_CODE = os.getenv("REQUIRE_CODE", "0") == "1"

def _h(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()

if REQUIRE_CODE:
    expected = _h(ACCESS_CODE)
    qp = st.experimental_get_query_params()
    unlocked = st.session_state.get("unlock_hash") == expected

    if (not unlocked) or (qp.get("locked") != ["0"]):
        st.set_page_config(page_title="Bayut & Dubizzle – Access Required", layout="centered")

        st.markdown(
            """
            <div style="max-width:420px;margin:120px auto;text-align:center;">
              <h2>Bayut & Dubizzle</h2>
              <p style="color:#666;">Internal AI Assistant</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        code = st.text_input(
            "Access code",
            type="password",
            placeholder="Enter access code",
            label_visibility="collapsed"
        )

        if st.button("Unlock"):
            if code == ACCESS_CODE:
                st.session_state["unlock_hash"] = expected
                st.experimental_set_query_params(locked="0")
                st.rerun()
            else:
                st.error("Wrong access code")

        st.stop()

# ===============================
# PAGE CONFIG (KEEP SIMPLE)
# ===============================
st.set_page_config(page_title="Bayut & Dubizzle Internal Assistant", layout="wide")

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===============================
# HF OFFLINE CACHE
# ===============================
MODEL_SRC = "/models"
MODEL_CACHE = "/tmp/models"

def _copy_baked_model():
    try:
        if os.path.isdir(MODEL_SRC):
            os.makedirs(MODEL_CACHE, exist_ok=True)
            if not os.listdir(MODEL_CACHE):
                for i in os.listdir(MODEL_SRC):
                    s = os.path.join(MODEL_SRC, i)
                    d = os.path.join(MODEL_CACHE, i)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
    except Exception:
        pass

_copy_baked_model()

os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("answer_mode", "Ultra-Fast")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# ===============================
# HELPERS
# ===============================
def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def clean_answer(text: str) -> str:
    """
    Return ONE clean answer (no dumping multiple Qs).
    - If chunk contains Q/A format -> extract the first A block only
    - Otherwise -> remove Q/A markers and cut at the next Q if it appears
    """
    if not text:
        return ""

    t = text.strip()

    # Extract first "A: ..." block if present
    m = re.search(r"\bA\s*[:–-]\s*(.*?)(?=\n\s*Q\d*\s*[:–-]|\Z)", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
    else:
        # Remove leading Q/A markers if present
        t = re.sub(r"^\s*Q\d*\s*[:–-]\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*A\s*[:–-]\s*", "", t, flags=re.IGNORECASE)

        # If another Q appears later, cut before it (prevents stitched dumps)
        t = re.split(r"\n\s*Q\d*\s*[:–-]\s*", t, maxsplit=1, flags=re.IGNORECASE)[0]
        t = re.split(r"\s+Q\d*\s*[:–-]\s*", t, maxsplit=1, flags=re.IGNORECASE)[0]

    # Normalize whitespace
    t = re.sub(r"\s{2,}", " ", t).strip()

    # Safety: remove any remaining standalone "A:" that leaks through
    t = re.sub(r"\bA\s*[:–-]\s*", "", t, flags=re.IGNORECASE).strip()

    return t

# ===============================
# EMBEDDINGS
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=MODEL_CACHE,
        model_kwargs={"local_files_only": True}
    )

# ===============================
# BUILD FAISS INDEXES (GENERAL + BAYUT + DUBIZZLE)
# ===============================
@st.cache_resource
def build_indexes():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    general_docs = []
    bayut_docs = []
    dubizzle_docs = []

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

        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": file})
            general_docs.append(doc)
            if bucket == "bayut":
                bayut_docs.append(doc)
            elif bucket == "dubizzle":
                dubizzle_docs.append(doc)

    if not general_docs:
        raise RuntimeError("❌ No readable .txt files found")

    emb = get_embeddings()

    vs_general = FAISS.from_documents(general_docs, emb)
    vs_bayut = FAISS.from_documents(bayut_docs, emb) if bayut_docs else None
    vs_dubizzle = FAISS.from_documents(dubizzle_docs, emb) if dubizzle_docs else None

    return vs_general, vs_bayut, vs_dubizzle

VS_GENERAL, VS_BAYUT, VS_DUBIZZLE = build_indexes()

def get_vectorstore(mode: str):
    if mode == "Bayut" and VS_BAYUT is not None:
        return VS_BAYUT
    if mode == "Dubizzle" and VS_DUBIZZLE is not None:
        return VS_DUBIZZLE
    return VS_GENERAL

# ===============================
# TITLE (KEEP SAME SIMPLE LOOK)
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;margin-bottom:0;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    <p style="text-align:center;color:#666;margin-top:6px;">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# MODE BUTTONS (CENTERED, SIMPLE)
# ===============================
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c2:
    if st.button("General", use_container_width=True):
        st.session_state.tool_mode = "General"
with c3:
    if st.button("Bayut", use_container_width=True):
        st.session_state.tool_mode = "Bayut"
with c4:
    if st.button("Dubizzle", use_container_width=True):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(f"<div style='text-align:center;font-size:18px;font-weight:700;margin-top:6px;'>{st.session_state.tool_mode} Assistant</div>", unsafe_allow_html=True)

# ===============================
# ANSWER MODE (SIMPLE, NO UI DRIFT)
# ===============================
m1, m2, m3 = st.columns([2, 1, 2])
with m2:
    st.session_state.answer_mode = st.radio(
        "Answer mode",
        ["Ultra-Fast", "Thinking"],
        index=0 if st.session_state.answer_mode == "Ultra-Fast" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# ===============================
# QUESTION INPUT (CENTERED)
# ===============================
left, mid, right = st.columns([1, 3, 1])
with mid:
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
        ask = st.form_submit_button("Ask", use_container_width=True)

# ===============================
# ANSWERING (ONE CLEAN ANSWER, SAME STYLE)
# ===============================
if ask and q:
    if st.session_state.answer_mode == "Thinking":
        with st.spinner("Thinking..."):
            time.sleep(0.8)

    vs = get_vectorstore(st.session_state.tool_mode)

    # IMPORTANT: k=1 so we NEVER dump mixed Qs
    results = vs.similarity_search(q, k=1)

    if results:
        answer = clean_answer(results[0].page_content)
        if not answer:
            answer = "No relevant information found in internal files."
    else:
        answer = "No relevant information found in internal files."

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY (SIMPLE)
# ===============================
history = st.session_state.chat[st.session_state.tool_mode]
for item in reversed(history):
    st.markdown(f"**Q:** {html.escape(item['q'])}")
    st.markdown(item["a"])
    st.markdown("---")
