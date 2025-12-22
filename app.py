import os
import re
import html
import time
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================
# 🔐 ACCESS CODE GATE (MUST BE FIRST – BEFORE ANY UI)
# =====================================================
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
REQUIRE_CODE = os.getenv("REQUIRE_CODE", "0") == "1"

if REQUIRE_CODE:
    st.session_state.setdefault("unlocked", False)

    if not st.session_state["unlocked"]:
        st.set_page_config(
            page_title="Bayut & Dubizzle – Access Required",
            layout="centered"
        )

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
            label_visibility="collapsed",
            placeholder="Enter access code"
        )

        if st.button("Unlock"):
            if ACCESS_CODE and code == ACCESS_CODE:
                st.session_state["unlocked"] = True
                st.rerun()
            else:
                st.error("Wrong access code")

        st.stop()


# ===============================
# PAGE CONFIG (MAIN APP)
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# UI CSS
# ===============================
st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { gap: 0.12rem; }
      button { white-space: nowrap !important; }

      .mode-title{
        font-size: 20px;
        font-weight: 750;
        margin: 6px 0;
      }

      .question-wrap{
        max-width: 980px;
        margin: 10px auto;
      }

      .question-wrap [data-testid="stForm"]{
        border: 1px solid #E7E9EE;
        border-radius: 12px;
        padding: 16px 18px 10px;
        background: #fff;
      }

      .question-wrap div[data-testid="stTextInput"] > label {
        display:none;
      }

      hr { margin: 14px 0; }
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
# OFFLINE HF MODEL
# ===============================
MODEL_SRC = "/models"
MODEL_CACHE = "/tmp/models"

def _copy_baked_model_to_tmp():
    try:
        if os.path.isdir(MODEL_SRC):
            os.makedirs(MODEL_CACHE, exist_ok=True)
            if not os.listdir(MODEL_CACHE):
                for item in os.listdir(MODEL_SRC):
                    s = os.path.join(MODEL_SRC, item)
                    d = os.path.join(MODEL_CACHE, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
    except Exception:
        pass

_copy_baked_model_to_tmp()

os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===============================
# SESSION STATE
# ===============================
if "chat" not in st.session_state or not isinstance(st.session_state.chat, dict):
    st.session_state.chat = {}

st.session_state.chat.setdefault("General", [])
st.session_state.chat.setdefault("Bayut", [])
st.session_state.chat.setdefault("Dubizzle", [])

THINKING_DELAY_SECONDS = 1.2

# ===============================
# FILE RULES
# ===============================
DOWNLOAD_ONLY_FILES = {
    "bayut-algolia locations sop.txt",
    "bayut-mybayut newsletters sop.txt",
    "bayut-pm campaigns sop.txt",
    "bayut-social media posting sop.txt",
    "both corrections and updates for listings.txt",
    "dubizzle newsletters sop.txt",
    "dubizzle pm campaigns sop.txt",
}

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

def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"\bQ:\s*|\bA:\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=MODEL_CACHE,
        model_kwargs={"local_files_only": True}
    )

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Select tool")
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"])
    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"])

chat_key = tool_mode
st.session_state.chat.setdefault(chat_key, [])

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
    col1, col2 = st.columns(2)
    ask = col1.form_submit_button("Ask")
    clear = col2.form_submit_button("Clear chat")

st.markdown("</div>", unsafe_allow_html=True)

if clear:
    st.session_state.chat[chat_key] = []
    st.rerun()

# ===============================
# HANDLE QUESTION (placeholder logic)
# ===============================
if ask and q:
    st.session_state.chat[chat_key].append({
        "q": q,
        "a": "Answer logic already wired correctly."
    })
    st.rerun()

# ===============================
# CHAT HISTORY
# ===============================
for item in reversed(st.session_state.chat[chat_key]):
    st.markdown(f"**Q:** {html.escape(item['q'])}")
    st.markdown(clean_answer(item["a"]))
    st.markdown("---")
