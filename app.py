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

    if not unlocked or qp.get("locked") != ["0"]:
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
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Bayut & Dubizzle Internal Assistant", layout="wide")

# ===============================
# UI CSS (KEEP SAME DESIGN)
# ===============================
st.markdown(
    """
    <style>
      .mode-title{font-size:20px;font-weight:700;margin:6px 0;}
      .question-wrap{max-width:980px;margin:10px auto;}
      .question-wrap [data-testid="stForm"]{
        border:1px solid #E7E9EE;border-radius:12px;padding:16px;background:#fff;
      }
      .question-wrap label{display:none;}
    </style>
    """,
    unsafe_allow_html=True
)

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
                    shutil.copytree(s, d, dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy2(s, d)
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
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# ===============================
# HELPERS
# ===============================
def read_text(fp):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def clean_answer(txt: str) -> str:
    """
    Clean the retrieved chunk so we output ONLY the answer text.
    - removes leading "Q:" / "A:" and similar variants
    - collapses extra whitespace
    - if a chunk contains multiple Q/A blocks, keep ONLY the first answer-like part
    """
    if not txt:
        return ""

    # Normalize lines
    t = txt.strip()

    # If the chunk contains multiple QA blocks, cut at the next "Q:" occurrence (after the first)
    # This prevents the "long dump" behavior.
    parts = re.split(r"\n\s*Q\d*\s*[:–-]\s*", t)
    t = parts[0] if parts else t

    # Remove inline Q/A markers
    t = re.sub(r"\bQ\d*\s*[:–-]?\s*", "", t)
    t = re.sub(r"\bA\s*[:–-]?\s*", "", t)

    # If there's still a " Q:" somewhere later, cut it (extra safety)
    t = re.split(r"\s+Q\d*\s*[:–-]\s*", t)[0]

    return re.sub(r"\s{2,}", " ", t).strip()

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
# 🔥 BUILD FAISS INDEX FROM /data
# ===============================
@st.cache_resource
def build_index():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    for file in os.listdir(DATA_DIR):
        if not file.lower().endswith(".txt"):
            continue

        raw = read_text(os.path.join(DATA_DIR, file))
        for chunk in splitter.split_text(raw):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": file}
                )
            )

    if not docs:
        raise RuntimeError("❌ No readable .txt files found")

    return FAISS.from_documents(docs, get_embeddings())

VECTORSTORE = build_index()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    tool_mode = st.radio("Select tool", ["General", "Bayut", "Dubizzle"])
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"])

# ===============================
# MAIN TITLE
# ===============================
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

# ===============================
# QUESTION UI
# ===============================
st.markdown('<div class="question-wrap">', unsafe_allow_html=True)
st.markdown(f'<div class="mode-title">{tool_mode} Assistant</div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here...")
    ask = st.form_submit_button("Ask")

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ANSWERING — SAME STYLE (ONE CLEAN ANSWER)
# ===============================
if ask and q:
    if answer_mode == "Thinking":
        with st.spinner("Thinking..."):
            time.sleep(1)

    # IMPORTANT: return only the single best chunk to avoid long mixed answers
    results = VECTORSTORE.similarity_search(q, k=1)

    if results:
        answer = clean_answer(results[0].page_content)
    else:
        answer = "No relevant information found in internal files."

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY
# ===============================
for item in reversed(st.session_state.chat[tool_mode]):
    st.markdown(f"**Q:** {html.escape(item['q'])}")
    st.markdown(item["a"])
    st.markdown("---")
