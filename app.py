import os
import re
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    answer_mode = st.radio("Answer mode", ["Short", "Detailed"], index=0)

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
# EMBEDDINGS (LOCAL — NO RATE LIMIT)
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===============================
# LLM
# ===============================
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# ===============================
# BUILD / LOAD INDEX
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
# QUESTION INTENT DETECTION
# ===============================
def is_definition_question(q: str) -> bool:
    q = q.lower().strip()
    return q.startswith("what is") or q.startswith("what are") or q.startswith("define")

def wants_process(q: str) -> bool:
    q = q.lower()
    keywords = ["how", "steps", "process", "workflow", "sop", "procedure"]
    return any(k in q for k in keywords)

def is_greeting(q: str) -> bool:
    return q.lower() in {"hi", "hello", "hey", "مرحبا", "اهلا"}

def is_app_question(q: str) -> bool:
    return "what is this app" in q.lower() or "what does this app do" in q.lower()

# ===============================
# PROMPT BUILDER (THE FIX)
# ===============================
def build_prompt(question: str, context: str, mode: str) -> str:
    if is_definition_question(question) and not wants_process(question):
        return f"""
You are an internal assistant.

Give a CLEAR, SHORT definition in 1–3 sentences.
Do NOT list steps.
Do NOT describe workflows.
Do NOT mention tools unless necessary.

Context:
{context}

Question:
{question}
"""

    if mode == "Short":
        return f"""
Answer briefly and clearly.
Summarize in plain language.
Avoid SOP-style details unless explicitly requested.

Context:
{context}

Question:
{question}
"""

    # Detailed
    return f"""
You are an internal SOP assistant.

Explain clearly and in a structured way.
Include steps only if relevant.
Be practical, not verbose.

Context:
{context}

Question:
{question}
"""

# ===============================
# ANSWERING
# ===============================
def answer_question(q: str, docs, mode: str):
    context = "\n\n".join(d.page_content for d in docs)[:3000]
    prompt = build_prompt(q, context, mode)
    return get_llm().invoke(prompt).content.strip()

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
        st.session_state.last_a = (
            "This is an internal AI assistant for Bayut & Dubizzle. "
            "It answers questions based on internal SOPs and guidelines."
        )
        st.rerun()

    docs = index.similarity_search(q_clean, k=4)
    ans = answer_question(q_clean, docs, answer_mode)

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
