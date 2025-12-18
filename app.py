import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =========================================
# SESSION STATE
# =========================================
if "last_q" not in st.session_state:
    st.session_state.last_q = ""
if "last_a" not in st.session_state:
    st.session_state.last_a = ""

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    mode = st.radio("Select module", ["General", "Bayut", "Dubizzle"])
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

# =========================================
# TITLE
# =========================================
st.markdown(
    """
    <h1 style="text-align:center;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

# =========================================
# PATHS
# =========================================
def get_paths(mode):
    base = "/tmp/faiss"
    if mode == "Bayut":
        return "data/bayut", f"{base}_bayut"
    elif mode == "Dubizzle":
        return "data/dubizzle", f"{base}_dubizzle"
    return "data/general", f"{base}_general"

DATA_DIR, INDEX_PATH = get_paths(mode)

# =========================================
# EMBEDDINGS (OPENAI – SAFE)
# =========================================
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# =========================================
# INDEX
# =========================================
@st.cache_resource
def load_or_build_index(data_dir, index_path):
    embeddings = get_embeddings()

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)

    if not os.path.exists(data_dir):
        return None

    docs = []
    for f in os.listdir(data_dir):
        if f.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, f), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index

index = load_or_build_index(DATA_DIR, INDEX_PATH)

# =========================================
# ULTRA FAST
# =========================================
def extractive_answer(question, docs):
    if not docs:
        return "No internal content found."

    text = docs[0].page_content
    sentences = re.split(r"(?<=[.!?])\s+", text)

    q_words = set(re.findall(r"\w+", question.lower()))
    scored = []

    for s in sentences:
        score = len(q_words & set(re.findall(r"\w+", s.lower())))
        scored.append((score, s))

    scored.sort(reverse=True)
    return " ".join(s for _, s in scored[:3])

# =========================================
# THINKING MODE
# =========================================
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

def thinking_answer(question, docs):
    if not docs:
        return "No internal content found."

    llm = get_llm()
    context = "\n".join(d.page_content for d in docs)[:2000]

    prompt = f"""
Use ONLY the context below.
Answer clearly.
Do not repeat the question.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content.strip()

# =========================================
# UI
# =========================================
st.subheader(f"Ask a {mode} question")

with st.form("ask", clear_on_submit=True):
    q = st.text_input("Question")
    ask = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

if clear:
    st.session_state.last_q = ""
    st.session_state.last_a = ""
    st.rerun()

if ask:
    if not q.strip():
        st.warning("Enter a question.")
    elif index is None:
        st.error("Index not available.")
    else:
        docs = index.similarity_search(q, k=2)

        if answer_mode == "Ultra-Fast":
            ans = extractive_answer(q, docs)
        else:
            with st.spinner("Thinking..."):
                ans = thinking_answer(q, docs)

        st.session_state.last_q = q
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
