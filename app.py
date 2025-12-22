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
REQUIRE_CODE = os.getenv("REQUIRE_CODE", "0") == "1"
ACCESS_CODE = os.getenv("ACCESS_CODE", "")

if REQUIRE_CODE:
    st.session_state.setdefault("unlocked", False)

    if not st.session_state.unlocked:
        st.set_page_config(
            page_title="Bayut & Dubizzle – Access Required",
            layout="centered"
        )

        st.markdown(
            """
            <style>
              .gate-wrap{
                max-width:420px;
                margin:120px auto;
                text-align:center;
                padding:24px 22px;
                border:1px solid #E7E9EE;
                border-radius:14px;
                background:#fff;
                box-shadow:0 6px 18px rgba(0,0,0,0.05);
              }
              .gate-title{ font-size:26px; font-weight:800; margin:0; }
              .gate-sub{ color:#666; margin-top:6px; margin-bottom:18px; }
              .gate-hint{ color:#888; font-size:12px; margin-top:10px; }
              /* center input a bit */
              div[data-testid="stTextInput"]{ max-width:340px; margin:0 auto; }
              div[data-testid="stButton"]{ display:flex; justify-content:center; }
            </style>

            <div class="gate-wrap">
              <div class="gate-title">
                <span style="color:#0E8A6D;">Bayut</span> &
                <span style="color:#D71920;">Dubizzle</span>
              </div>
              <div class="gate-sub">Internal AI Assistant</div>
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
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Wrong access code")

        # Optional tiny debug (won't leak the code)
        # st.caption(f"REQUIRE_CODE={os.getenv('REQUIRE_CODE')} | ACCESS_CODE={'SET' if ACCESS_CODE else 'MISSING'}")

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
# If your Dockerfile preloads the model into /models, we copy it into /tmp/models (writable at runtime).
MODEL_SRC = "/models"
MODEL_CACHE = "/tmp/models"

def _copy_baked_model_to_tmp():
    try:
        if os.path.isdir(MODEL_SRC):
            os.makedirs(MODEL_CACHE, exist_ok=True)
            # Only copy once if tmp cache is empty
            if not os.listdir(MODEL_CACHE):
                for item in os.listdir(MODEL_SRC):
                    s = os.path.join(MODEL_SRC, item)
                    d = os.path.join(MODEL_CACHE, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
    except Exception:
        # Do not crash UI because of copying
        pass

_copy_baked_model_to_tmp()

# Force all HF libs to use writable cache + offline mode
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

# migrate legacy key: "dubizzle" -> "Dubizzle" (extra safety)
if "dubizzle" in st.session_state.chat and "Dubizzle" not in st.session_state.chat:
    st.session_state.chat["Dubizzle"] = st.session_state.chat.pop("dubizzle")

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
def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\bQ:\s*|\bA:\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "Dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

def list_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".txt")
    )

@st.cache_resource
def get_embeddings():
    # Offline-only: will fail if model wasn't cached in image or copied successfully
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=MODEL_CACHE,
        model_kwargs={"local_files_only": True}
    )

# ===============================
# (OPTIONAL) INDEX BUILD PLACEHOLDERS
# ===============================
@st.cache_resource
def load_placeholder_index():
    # Keep imports used and prevent “unused” logic issues; replace with your real indexing if needed.
    emb = get_embeddings()
    docs = [Document(page_content="hello", metadata={"answer": "hi"})]
    return FAISS.from_documents(docs, emb)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Select tool")
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)
    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

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

if tool_mode == "Bayut":
    st.markdown(
        '<div class="mode-title">Ask <span style="color:#0E8A6D;font-weight:800;">Bayut</span> Anything</div>',
        unsafe_allow_html=True
    )
elif tool_mode == "Dubizzle":
    st.markdown(
        '<div class="mode-title">Ask <span style="color:#D71920;font-weight:800;">Dubizzle</span> Anything</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="mode-title">General Assistant</div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here...")

    left, mid, right = st.columns([4, 2.3, 4])
    with mid:
        b1, b2 = st.columns([1, 1], gap="small")
        ask = b1.form_submit_button("Ask")
        clear = b2.form_submit_button("Clear chat")

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# CLEAR CHAT
# ===============================
if clear:
    st.session_state.chat[chat_key] = []
    st.rerun()

# ===============================
# HANDLE QUESTION (PLACEHOLDER ANSWERING)
# ===============================
def _fake_answer(question: str) -> str:
    # Replace with your real logic
    return "Answer logic already wired correctly."

if ask and (q or "").strip():
    if answer_mode == "Thinking":
        with st.spinner("Thinking..."):
            time.sleep(THINKING_DELAY_SECONDS)

    st.session_state.chat[chat_key].append({"q": q, "a": _fake_answer(q)})
    st.rerun()

# ===============================
# CHAT HISTORY (NEWEST ON TOP)
# ===============================
style = bubble_style(chat_key)
items = list(enumerate(st.session_state.chat.get(chat_key, [])))[::-1]

for orig_idx, item in items:
    q_txt = html.escape(item.get("q", ""))

    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:6px;">
            <b>Q:</b> {q_txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    a_txt = clean_answer(item.get("a", ""))
    if a_txt:
        st.markdown(a_txt)

    st.markdown("---")
