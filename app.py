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
TMP_DIR = "/tmp"

# ===============================
# SESSION STATE (PER TOOL MEMORY)
# ===============================
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "Dubizzle": []
})
st.session_state.setdefault("topic", {
    "General": "",
    "Bayut": "",
    "Dubizzle": ""
})

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Tool")
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)

    st.markdown(" ")
    st.header("Answer mode")
    answer_mode = st.radio("", ["Ultra-Fast", "Thinking"], index=0)

    st.markdown(" ")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat[tool_mode] = []
        st.session_state.topic[tool_mode] = ""
        st.rerun()

    if st.button("🔁 Rebuild Index"):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        st.cache_resource.clear()
        st.success("Indexes cleared.")

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
# EMBEDDINGS
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===============================
# LLM (THINKING MODE)
# ===============================
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===============================
# FILE ROUTING
# ===============================
def file_allowed(filename: str, mode: str) -> bool:
    f = filename.lower()
    if mode == "General":
        return True
    if mode == "Bayut":
        return f.startswith("bayut") or f.startswith("both")
    if mode == "Dubizzle":
        return f.startswith("dubizzle") or f.startswith("both")
    return True

# ===============================
# INDEX PER TOOL
# ===============================
@st.cache_resource
def load_index(mode: str):
    index_path = os.path.join(TMP_DIR, f"faiss_{mode.lower()}")
    embeddings = get_embeddings()

    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt") and file_allowed(f, mode):
            docs.extend(
                TextLoader(os.path.join(DATA_DIR, f), encoding="utf-8").load()
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index

index = load_index(tool_mode)

# ===============================
# SMART CLEANING
# ===============================
def clean_query(q: str) -> str:
    q = re.sub(r"\blunch\b", "launch", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcampains\b", "campaigns", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpm\b", "paid marketing", q, flags=re.IGNORECASE)
    return q.strip()

# ===============================
# SMART ANSWER LAYER (THE FIX)
# ===============================
def smart_definition_answer(question, docs):
    """
    Produces a short, logical, human answer.
    """
    if not docs:
        return "I couldn’t find a clear definition in the available documentation."

    text = " ".join(d.page_content for d in docs)[:1200]

    prompt = f"""
Explain this clearly in 2–3 sentences.
Do NOT copy SOP text.
Explain like answering a colleague.

Topic:
{question}

Reference:
{text}

Answer:
"""
    return get_llm().invoke(prompt).content.strip()

# ===============================
# UI
# ===============================
st.subheader("Ask your internal question")
q = st.text_input("Question")

if st.button("Ask"):
    q_clean = clean_query(q)

    # Resolve context per tool
    topic = st.session_state.topic[tool_mode]
    resolved_query = f"{topic}. {q_clean}" if topic else q_clean

    docs = index.similarity_search(resolved_query, k=4)

    if answer_mode == "Ultra-Fast":
        ans = smart_definition_answer(q_clean, docs)
    else:
        ans = smart_definition_answer(q_clean, docs)

    # Update memory per tool
    st.session_state.topic[tool_mode] = q_clean
    st.session_state.chat[tool_mode].append({
        "q": q_clean,
        "a": ans
    })

# ===============================
# CHAT HISTORY (PER TOOL)
# ===============================
for item in st.session_state.chat[tool_mode]:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:8px;">
          <b>{item['q']}</b><br>
          {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )
