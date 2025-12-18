import os
import re
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
# SESSION STATE
# ===============================
st.session_state.setdefault("last_q", "")
st.session_state.setdefault("last_a", "")

# ===============================
# SIDEBAR (VISUAL ONLY)
# ===============================
with st.sidebar:
    st.header("Select an option")
    st.radio("", ["General", "Bayut", "Dubizzle"], index=0)
    st.markdown(" ")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

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
# SINGLE DATA SOURCE (NO MODULES)
# ===============================
DATA_DIR = "data/general"
INDEX_PATH = "/tmp/faiss_general"

# ===============================
# EMBEDDINGS — OPENAI ONLY
# ===============================
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ===============================
# INDEX
# ===============================
@st.cache_resource
def load_or_build_index():
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    if not os.path.exists(DATA_DIR):
        return None

    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            docs.extend(
                TextLoader(
                    os.path.join(DATA_DIR, f),
                    encoding="utf-8"
                ).load()
            )

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index

index = load_or_build_index()

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
    return " ".join(ranked[:3])

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def thinking_answer(q, docs):
    ctx = "\n".join(d.page_content for d in docs)[:2000]
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
    if not q.strip():
        st.warning("Enter a question.")
    elif index is None:
        st.error("Index not available.")
    else:
        docs = index.similarity_search(q, k=2)
        ans = (
            extractive_answer(q, docs)
            if answer_mode == "Ultra-Fast"
            else thinking_answer(q, docs)
        )
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
