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
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR = "/tmp"

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
        try:
            st.cache_resource.clear()
        except Exception:
            pass

        for p in ["general", "bayut", "dubizzle"]:
            ip = os.path.join(TMP_DIR, f"faiss_{p}")
            if os.path.exists(ip):
                shutil.rmtree(ip, ignore_errors=True)

        st.success("Index cache cleared. It will rebuild on the next question.")

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
# FILE ROUTING BY FILENAME
# ===============================
def classify_file(fname: str) -> str:
    f = fname.lower().strip()

    # Both
    if f.startswith("both") or " both " in f:
        return "both"

    # Bayut
    if f.startswith("bayut") or "bayut" in f:
        return "bayut"

    # Dubizzle
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
    def _sleep(self, attempt: int):
        wait = min(60, (2 ** attempt)) + random.uniform(0.2, 1.2)
        time.sleep(wait)

    def embed_documents(self, texts):
        # IMPORTANT: skip empty strings
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []

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
                    time.sleep(1.0)
                    if attempt == 7:
                        raise
        return out

    def embed_query(self, text):
        text = text or ""
        if not text.strip():
            return super().embed_query(" ")  # never empty
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
    return SafeOpenAIEmbeddings(
        model="text-embedding-3-small",
        max_retries=2,
        request_timeout=60
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=60,
        max_retries=2
    )

# ===============================
# SMART QUERY EXPANSION
# ===============================
def expand_query(q: str) -> str:
    x = q.strip()

    # fix typos
    x = re.sub(r"\blunch\b", "launch", x, flags=re.IGNORECASE)
    x = re.sub(r"\bcampains\b", "campaigns", x, flags=re.IGNORECASE)

    # org-specific expansions
    x = re.sub(r"\bpm\b", "paid marketing", x, flags=re.IGNORECASE)

    boosters = ["SOP", "process", "steps", "timeline", "schedule", "launch date", "go live", "start date"]
    return f"{x}\n\nKeywords: {', '.join(boosters)}"

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
        "what can you do", "who are you", "how to use", "help", "what is this"
    ]
    return any(t in x for t in triggers)

def app_description(mode: str) -> str:
    scope = {
        "General": "Bayut + Dubizzle + shared SOPs",
        "Bayut": "Bayut SOPs + Both SOPs",
        "Dubizzle": "Dubizzle SOPs + Both SOPs",
    }.get(mode, "Bayut + Dubizzle")

    return (
        "This is an internal AI content assistant for Bayut & Dubizzle.\n\n"
        f"- It searches your uploaded SOP .txt files and answers based on them.\n"
        f"- Current mode: **{mode}** → scope: **{scope}**.\n\n"
        "Ask naturally like you would to a colleague, e.g.:\n"
        "- “When do PM campaigns launch?”\n"
        "- “What’s the SOP for newsletters?”"
    )

# ===============================
# SAFE DOCUMENT LOADING + FILTERING
# ===============================
MIN_CHUNK_CHARS = 40

def safe_load_txt(file_path: str):
    """Return list of Documents, or [] if file is empty/unreadable."""
    try:
        docs = TextLoader(file_path, encoding="utf-8").load()
    except Exception:
        try:
            docs = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True).load()
        except Exception:
            return []

    # Filter documents with no real text
    cleaned = []
    for d in docs:
        if getattr(d, "page_content", "") and d.page_content.strip():
            cleaned.append(d)
    return cleaned

def filter_chunks(chunks):
    """Remove empty/too-short chunks (prevents FAISS embedding empty lists)."""
    out = []
    for c in chunks:
        txt = getattr(c, "page_content", "")
        if isinstance(txt, str) and len(txt.strip()) >= MIN_CHUNK_CHARS:
            out.append(c)
    return out

# ===============================
# INDEX BUILD/LOAD (PER MODE) — CRASH-PROOF
# ===============================
def load_or_build_index(mode: str):
    embeddings = get_embeddings()
    mode_key = mode.lower()
    index_path = os.path.join(TMP_DIR, f"faiss_{mode_key}")

    # load existing
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            # if corrupted, rebuild cleanly
            shutil.rmtree(index_path, ignore_errors=True)

    if not os.path.exists(DATA_DIR):
        st.error(f"Data folder not found: `{DATA_DIR}`")
        return None

    # Load docs
    docs = []
    used_files = 0
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt"):
            continue
        fclass = classify_file(f)
        if not allowed_in_mode(fclass, mode):
            continue

        fp = os.path.join(DATA_DIR, f)
        loaded = safe_load_txt(fp)
        if loaded:
            docs.extend(loaded)
            used_files += 1

    if not docs:
        st.error(
            f"No usable text found for **{mode}**.\n\n"
            "✅ Make sure your .txt files are NOT empty and contain real text."
        )
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    chunks = filter_chunks(chunks)

    # CRITICAL: never build FAISS with empty chunks
    if not chunks:
        st.error(
            f"I loaded **{used_files}** file(s) for **{mode}**, but all content was empty/too short after cleaning.\n\n"
            "Fix: open your .txt files and make sure they contain real text (not just titles/spaces)."
        )
        return None

    texts = [c.page_content for c in chunks if c.page_content and c.page_content.strip()]
    metadatas = [getattr(c, "metadata", {}) for c in chunks]

    if not texts:
        st.error("No valid text chunks to embed after filtering. Check your .txt content.")
        return None

    prog = st.progress(0, text="Building index (first time only)...")
    try:
        # Build from texts (more controlled than from_documents)
        index = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        index.save_local(index_path)
        prog.progress(100, text="Index built ✅")
        return index

    except openai.RateLimitError:
        shutil.rmtree(index_path, ignore_errors=True)
        st.error("OpenAI rate limit hit while building the index. Wait 1–2 minutes, then try again.")
        return None

    except Exception as e:
        shutil.rmtree(index_path, ignore_errors=True)
        st.error(f"Index build failed: {type(e).__name__}. This usually means empty/invalid file content.")
        return None

    finally:
        try:
            time.sleep(0.2)
            prog.empty()
        except Exception:
            pass

# ===============================
# ANSWERING (CRASH-PROOF)
# ===============================
def extractive_answer(q, docs):
    if not docs:
        return "I couldn’t find anything relevant in the uploaded SOPs."

    combined = "\n".join((d.page_content or "") for d in docs)
    combined = combined.strip()
    if not combined:
        return "I found related files, but the extracted text was empty."

    sentences = re.split(r"(?<=[.!?])\s+", combined)
    q_words = set(re.findall(r"\w+", (q or "").lower()))

    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True
    )
    ranked = [s for s in ranked if s.strip()]
    return " ".join(ranked[:4]).strip() if ranked else "I couldn’t find a clear answer in the SOP text."

def thinking_answer(q, docs, mode):
    prefix = "General: "
    if mode == "Bayut":
        prefix = "Bayut: "
    elif mode == "Dubizzle":
        prefix = "Dubizzle: "

    if not docs:
        return prefix + "I couldn’t find anything relevant in the uploaded SOPs."

    ctx = "\n\n".join((d.page_content or "") for d in docs).strip()[:2800]
    if not ctx:
        return prefix + "I found related files, but the extracted context was empty."

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
        return prefix + extractive_answer(q, docs)

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

    # greetings
    if is_greeting(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = "Hello 👋 How can I help you?"
        st.rerun()

    # app question
    if is_app_question(q_clean):
        st.session_state.last_q = q_clean
        st.session_state.last_a = app_description(tool_mode)
        st.rerun()

    # retrieval
    search_q = expand_query(q_clean)

    index = load_or_build_index(tool_mode)
    if index is None:
        st.stop()

    docs = index.similarity_search(search_q, k=4)

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
