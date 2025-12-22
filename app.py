import os
import re
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🔑 LOCAL embeddings (NO OpenAI, NO rate limits)
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = "/tmp/faiss_index"

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("last_q", "")
st.session_state.setdefault("last_a", "")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Select an option")
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)

    st.markdown(" ")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

    st.markdown(" ")
    if st.button("Rebuild Index"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH, ignore_errors=True)
        st.cache_resource.clear()
        st.success("Index cleared. It will rebuild once.")

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    <p style="text-align:center;color:#666;">Fast internal assistant</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOCAL EMBEDDINGS (FINAL FIX)
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===============================
# BUILD / LOAD INDEX (FAST, SAFE)
# ===============================
@st.cache_resource
def load_index_once():
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".txt"):
            docs.extend(
                TextLoader(os.path.join(DATA_DIR, f), encoding="utf-8").load()
            )

    if not docs:
        st.error("No .txt files found in /data")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index

index = load_index_once()

# ===============================
# SMART QUERY CLEANING
# ===============================
def expand_query(q: str) -> str:
    x = q.strip()
    x = re.sub(r"\blunch\b", "launch", x, flags=re.IGNORECASE)
    x = re.sub(r"\bcampains\b", "campaigns", x, flags=re.IGNORECASE)
    x = re.sub(r"\bpm\b", "paid marketing", x, flags=re.IGNORECASE)
    return x

# ===============================
# SMALL TALK
# ===============================
def is_greeting(q: str):
    return q.lower() in {"hi", "hello", "hey", "مرحبا", "اهلا"}

def is_app_question(q: str):
    return "what is this" in q.lower() or "what does this app" in q.lower()

def app_description():
    return (
        "This is an internal AI assistant for Bayut & Dubizzle.\n\n"
        "It searches internal SOPs and answers questions naturally.\n\n"
        "Examples:\n"
        "- When do PM campaigns launch?\n"
        "- What’s the SOP for newsletters?"
    )

# ===============================
# ANSWERING
# ===============================
def extractive_answer(q, docs):
    text = "\n".join(d.page_content for d in docs)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    q_words = set(re.findall(r"\w+", q.lower()))
    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True
    )
    return " ".join(ranked[:4])

def thinking_answer(q, docs):
    ctx = "\n\n".join(d.page_content for d in docs)[:2500]
    return get_llm().invoke(
        f"Context:\n{ctx}\n\nQuestion:\n{q}"
    ).content.strip()

# ===============================
# UI
# ===============================
st.subheader("Ask your internal question")

with st.form("ask", clear_on_submit=True):
    q = st.text_input("Question")
    ask = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

if clear:
    st.session_state.last_q = ""
    st.session_state.last_a = ""
    st.rerun()

if ask:
    q_clean = q.strip()

    if not q_clean:
        st.warning("Enter a question.")
        st.stop()

    if is_greeting(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = "Hello 👋 How can I help you?"
        st.rerun()

    if is_app_question(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = app_description()
        st.rerun()

    search_q = expand_query(q_clean)
    docs = index.similarity_search(search_q, k=4)

    ans = (
        extractive_answer(q_clean, docs)
        if answer_mode == "Ultra-Fast"
        else thinking_answer(q_clean, docs)
    )

    st.session_state.last_q = q_clean
    st.session_state.last_a = ans
    st.rerun()

if st.session_state.last_q:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:16px;border-radius:8px;">
          <b>{st.session_state.last_q}</b><br><br>
          {st.session_state.last_a}
        </div>
        """,
        unsafe_allow_html=True
    )
