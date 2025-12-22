import os
import re
import time
import random
import shutil
import streamlit as st

import openai  # for catching RateLimitError

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
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")      # repo folder: /app/data
TMP_DIR = "/tmp"                               # writable on Cloud Run/Streamlit Cloud


# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("last_q", "")
st.session_state.setdefault("last_a", "")
st.session_state.setdefault("last_tool", "General")


# ===============================
# SIDEBAR
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

        st.success("Index cache cleared. It will rebuild automatically on next question.")


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
# SMART ROUTING BY FILENAME
# ===============================
def classify_file(fname: str) -> str:
    f = fname.lower().strip()

    # BOTH
    if f.startswith("both") or " both " in f:
        return "both"

    # BAYUT
    if f.startswith("bayut") or "bayut" in f:
        return "bayut"

    # DUBIZZLE
    if f.startswith("dubizzle") or "dubizzle" in f:
        return "dubizzle"

    return "general"


def allowed_in_mode(file_class: str, mode: str) -> bool:
    if mode == "General":
        return True
    if mode == "Bayut":
        return file_class in {"bayut", "both"}
    if mode == "Dubizzle":
        return file_class in {"dubizzle", "both"}
    return True


# ===============================
# RATE-LIMIT SAFE EMBEDDINGS
# ===============================
class SafeOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Wrap OpenAI embeddings to:
    - embed in small batches
    - retry on RateLimitError with exponential backoff
    """
    def _sleep(self, attempt: int):
        # exponential backoff + jitter
        wait = min(60, (2 ** attempt)) + random.uniform(0.2, 1.2)
        time.sleep(wait)

    def embed_documents(self, texts):
        # small batches to reduce bursts
        batch_size = 24
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(8):
                try:
                    out.extend(super().embed_documents(batch))
                    break
                except openai.RateLimitError:
                    self._sleep(attempt)
                except Exception:
                    # short pause for transient errors
                    time.sleep(1.0)
                    if attempt == 7:
                        raise
        return out

    def embed_query(self, text):
        for attempt in range(8):
            try:
                return super().embed_query(text)
            except openai.RateLimitError:
                self._sleep(attempt)
            except Exception:
                time.sleep(1.0)
                if attempt == 7:
                    raise


@st.cache_resource
def get_embeddings():
    # max_retries here is still useful, but our wrapper is the real fix
    return SafeOpenAIEmbeddings(
        model="text-embedding-3-small",
        max_retries=2,
        request_timeout=60
    )


# ===============================
# LLM
# ===============================
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=60, max_retries=2)


# ===============================
# SMART QUERY EXPANSION
# ===============================
def expand_query(q: str) -> str:
    x = q.strip()

    # Fix common typos
    x = re.sub(r"\blunch\b", "launch", x, flags=re.IGNORECASE)
    x = re.sub(r"\bcampains\b", "campaigns", x, flags=re.IGNORECASE)

    # Expand abbreviations / intent words
    # PM in your org context = Paid Marketing
    x2 = x
    x2 = re.sub(r"\bpm\b", "paid marketing", x2, flags=re.IGNORECASE)

    # Add extra keywords to help matching SOP phrasing
    boosters = [
        "schedule", "timeline", "when", "launch date", "process", "SOP"
    ]

    # If the question seems about campaigns/launching, boost those terms
    if re.search(r"\bcampaign|launch|paid marketing\b", x2, flags=re.IGNORECASE):
        boosters += ["campaign launch", "go live", "start date"]

    return f"{x2}\n\nKeywords: {', '.join(boosters)}"


# ===============================
# SMALLTALK + APP DESCRIPTION (NO SEARCH)
# ===============================
def is_greeting(q: str) -> bool:
    x = q.strip().lower()
    return (
        x in {"hi", "hello", "hey", "السلام عليكم", "سلام", "مرحبا", "أهلا", "اهلا"} or
        x.startswith(("hi ", "hello ", "hey ", "مرحبا "))
    )

def is_app_question(q: str) -> bool:
    x = q.strip().lower()
    triggers = [
        "what is this app", "what is this tool", "what does this app do",
        "what can you do", "who are you", "how to use", "help"
    ]
    return any(t in x for t in triggers)

def app_description(mode: str) -> str:
    scope = {
        "General": "Bayut + Dubizzle + shared SOPs",
        "Bayut": "Bayut SOPs + shared (Both) SOPs",
        "Dubizzle": "Dubizzle SOPs + shared (Both) SOPs",
    }.get(mode, "Bayut + Dubizzle")

    return (
        "This is an internal AI content assistant for Bayut & Dubizzle.\n\n"
        f"- It searches your uploaded SOP .txt files and answers based on them.\n"
        f"- Current mode: **{mode}** → scope: **{scope}**.\n\n"
        "Try questions like:\n"
        "- “When do paid marketing campaigns launch?”\n"
        "- “What’s the process for newsletters?”\n"
        "- “How do we handle listing corrections and updates?”"
    )


# ===============================
# INDEX LOAD/BUILD (PER MODE) — NOT CACHED
# ===============================
def load_or_build_index(mode: str):
    embeddings = get_embeddings()
    mode_key = mode.lower()
    index_path = os.path.join(TMP_DIR, f"faiss_{mode_key}")

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(DATA_DIR):
        return None

    docs = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt"):
            continue

        fclass = classify_file(f)
        if not allowed_in_mode(fclass, mode):
            continue

        fp = os.path.join(DATA_DIR, f)
        try:
            docs.extend(TextLoader(fp, encoding="utf-8").load())
        except Exception:
            docs.extend(TextLoader(fp, encoding="utf-8", autodetect_encoding=True).load())

    if not docs:
        return None

    # Fewer chunks => fewer embedding calls => fewer rate limit hits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # bigger chunks than before
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # Build with a progress UI so the user doesn't just see a crash
    prog = st.progress(0, text="Building index (first time only)...")
    try:
        # FAISS.from_documents will call our SafeOpenAIEmbeddings which retries safely
        index = FAISS.from_documents(chunks, embeddings)
        index.save_local(index_path)
        prog.progress(100, text="Index built ✅")
        return index
    except openai.RateLimitError:
        # Clean partial index to avoid corrupt cache
        if os.path.exists(index_path):
            shutil.rmtree(index_path, ignore_errors=True)
        prog.empty()
        st.error(
            "OpenAI rate limit hit while building the index.\n\n"
            "Try again in a minute, or press **Rebuild Index** once and retry."
        )
        return None
    finally:
        try:
            time.sleep(0.2)
            prog.empty()
        except Exception:
            pass


# ===============================
# ANSWERING
# ===============================
def extractive_answer(q, docs):
    # Use top docs, not just 1
    combined = "\n".join(d.page_content for d in docs)
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    q_words = set(re.findall(r"\w+", q.lower()))

    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True
    )
    return " ".join(ranked[:4]).strip()


def thinking_answer(q, docs, mode):
    ctx = "\n\n".join(d.page_content for d in docs)[:2800]

    prefix = "General: "
    if mode == "Bayut":
        prefix = "Bayut: "
    elif mode == "Dubizzle":
        prefix = "Dubizzle: "

    prompt = (
        "You are an internal SOP assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the context doesn't contain the answer, say you don't have enough info in the uploaded SOPs.\n"
        "Be practical and direct.\n\n"
        f"Mode: {mode}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{q}\n\n"
        f"Return the final answer. Start with '{prefix}'."
    )

    try:
        return get_llm().invoke(prompt).content.strip()
    except openai.RateLimitError:
        # fallback instead of crashing
        return (prefix + extractive_answer(q, docs))


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

    # Smalltalk (no search)
    if is_greeting(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = "Hello 👋 How can I help you?"
        st.rerun()

    # App description (no search)
    if is_app_question(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = app_description(tool_mode)
        st.rerun()

    # Smart retrieval query
    search_q = expand_query(q_clean)

    index = load_or_build_index(tool_mode)
    if index is None:
        st.stop()

    # More docs to handle “not direct like filenames”
    docs = index.similarity_search(search_q, k=4)

    if not docs:
        st.session_state.last_q = q_clean
        st.session_state.last_a = "I couldn’t find anything relevant in the uploaded SOPs for this mode."
        st.rerun()

    ans = extractive_answer(q_clean, docs) if answer_mode == "Ultra-Fast" else thinking_answer(q_clean, docs, tool_mode)

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
