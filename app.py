import os
import re
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# ===============================
# PATHS (works on Cloud Run / local)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # your repo /app/data
TMP_DIR = "/tmp"  # Cloud Run writable

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("last_q", "")
st.session_state.setdefault("last_a", "")
st.session_state.setdefault("last_tool", "General")

# ===============================
# SIDEBAR (VISUAL + MODE)
# ===============================
with st.sidebar:
    st.header("Select an option")

    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)
    st.session_state.last_tool = tool_mode

    st.markdown(" ")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

    st.markdown(" ")
    if st.button("Rebuild Index"):
        # Clear Streamlit cache + remove saved FAISS folders
        try:
            st.cache_resource.clear()
        except Exception:
            pass

        for p in ["general", "bayut", "dubizzle"]:
            ip = os.path.join(TMP_DIR, f"faiss_{p}")
            if os.path.exists(ip):
                shutil.rmtree(ip, ignore_errors=True)

        st.success("Rebuilding will happen automatically on next question.")

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
# EMBEDDINGS — OPENAI ONLY
# ===============================
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ===============================
# FILE ROUTING RULES (NO FOLDER CHANGES NEEDED)
# ===============================
def classify_file(fname: str) -> str:
    """
    Returns: "bayut" | "dubizzle" | "both" | "general"
    Based on filename patterns. Your current filenames already match this.
    """
    f = fname.lower().strip()

    # both
    if f.startswith("both ") or f.startswith("both-") or " both " in f or f.startswith("both_") or f.startswith("both"):
        return "both"

    # bayut
    if f.startswith("bayut") or f.startswith("bayut-") or "bayut" in f:
        return "bayut"

    # dubizzle
    if f.startswith("dubizzle") or f.startswith("dubizzle-") or "dubizzle" in f:
        return "dubizzle"

    return "general"

def allowed_in_mode(file_class: str, mode: str) -> bool:
    """
    General: everything
    Bayut: bayut + both
    Dubizzle: dubizzle + both
    """
    if mode == "General":
        return True
    if mode == "Bayut":
        return file_class in {"bayut", "both"}
    if mode == "Dubizzle":
        return file_class in {"dubizzle", "both"}
    return True

# ===============================
# INDEX BUILD/LOAD (PER MODE)
# ===============================
@st.cache_resource
def load_or_build_index(mode: str):
    """
    Builds a separate FAISS index per mode (general/bayut/dubizzle),
    but WITHOUT changing your folder structure.
    """
    embeddings = get_embeddings()

    mode_key = mode.lower()
    index_path = os.path.join(TMP_DIR, f"faiss_{mode_key}")

    # load if exists
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # validate data dir
    if not os.path.exists(DATA_DIR):
        return None

    # load only relevant docs
    docs = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt"):
            continue

        fclass = classify_file(f)
        if not allowed_in_mode(fclass, mode):
            continue

        file_path = os.path.join(DATA_DIR, f)
        try:
            docs.extend(TextLoader(file_path, encoding="utf-8").load())
        except Exception:
            # fallback if any weird encoding
            docs.extend(TextLoader(file_path, encoding="utf-8", autodetect_encoding=True).load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index

# ===============================
# SMALLTALK + APP DESCRIPTION (NO SEARCH)
# ===============================
def is_greeting(q: str) -> bool:
    x = q.strip().lower()
    return x in {"hi", "hello", "hey", "hiya", "السلام عليكم", "سلام", "مرحبا", "أهلا", "اهلا"} or x.startswith(("hi ", "hello ", "hey "))

def is_app_question(q: str) -> bool:
    x = q.strip().lower()
    patterns = [
        "what is this app", "what is this tool", "what is this", "what does this app do",
        "what can you do", "who are you", "help", "how to use"
    ]
    return any(p in x for p in patterns)

def app_description(mode: str) -> str:
    scope = {
        "General": "Bayut + Dubizzle (and shared SOPs)",
        "Bayut": "Bayut SOPs only (plus shared SOPs marked as Both)",
        "Dubizzle": "Dubizzle SOPs only (plus shared SOPs marked as Both)"
    }.get(mode, "Bayut + Dubizzle")

    return (
        f"This is an internal AI assistant for Bayut & Dubizzle.\n\n"
        f"- It searches your uploaded .txt SOPs and internal notes, then answers quickly.\n"
        f"- Current mode: **{mode}** → answering from **{scope}**.\n\n"
        f"Ask me anything like: “What is the process for X?” or “Where do we update Y?”"
    )

# ===============================
# ANSWERING
# ===============================
def extractive_answer(q, docs):
    text = docs[0].page_content
    sentences = re.split(r"(?<=[.!?])\s+", text)
    q_words = set(re.findall(r"\w+", q.lower()))

    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True
    )
    return " ".join(ranked[:3]).strip()

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def thinking_answer(q, docs, mode):
    ctx = "\n\n".join(d.page_content for d in docs)[:2500]

    system = (
        "You are an internal content operations assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the context does not contain the answer, say you don't have enough info in the uploaded SOPs.\n"
        "Be clear and practical."
    )

    prefix = ""
    if mode == "Bayut":
        prefix = "Bayut: "
    elif mode == "Dubizzle":
        prefix = "Dubizzle: "
    else:
        prefix = "General: "

    msg = (
        f"{system}\n\n"
        f"Mode: {mode}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{q}\n\n"
        f"Return the final answer. Start with '{prefix}'."
    )

    return get_llm().invoke(msg).content.strip()

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
    q_clean = (q or "").strip()

    if not q_clean:
        st.warning("Enter a question.")
        st.stop()

    # 1) Smalltalk / app description (NO SEARCH)
    if is_greeting(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = "Hello 👋 How can I help you?"
        st.rerun()

    if is_app_question(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = app_description(tool_mode)
        st.rerun()

    # 2) Retrieval-based answering (SEARCH)
    index = load_or_build_index(tool_mode)
    if index is None:
        st.error(
            f"No index available for **{tool_mode}**.\n\n"
            f"Make sure you have .txt files inside: `{DATA_DIR}`"
        )
        st.stop()

    docs = index.similarity_search(q_clean, k=2)

    if not docs:
        st.session_state.last_q = q_clean
        st.session_state.last_a = "I couldn’t find anything relevant in the uploaded SOPs for this mode."
        st.rerun()

    ans = (
        extractive_answer(q_clean, docs)
        if answer_mode == "Ultra-Fast"
        else thinking_answer(q_clean, docs, tool_mode)
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
