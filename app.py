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
    page_title="Bayut & dubizzle AI Assistant",
    layout="wide"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = "/tmp/qa_index"

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "dubizzle": []
})

# ===============================
# SIDEBAR – TOOL SELECTION
# ===============================
with st.sidebar:
    st.header("Select tool")
    tool_mode = st.radio("", ["General", "Bayut", "dubizzle"], index=0)

    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">dubizzle</span>
      AI Content Assistant
    </h1>
    <p style="text-align:center;color:#666;">Internal Q&A Assistant</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOAD QA FILES ONLY
# ===============================
def get_qa_files(mode):
    files = []
    for f in os.listdir(DATA_DIR):
        if not f.endswith("-QA.txt"):
            continue

        lf = f.lower()
        if mode == "Bayut" and not lf.startswith("bayut"):
            continue
        if mode == "dubizzle" and not lf.startswith("dubizzle"):
            continue

        files.append(os.path.join(DATA_DIR, f))
    return files

# ===============================
# BUILD / LOAD INDEX
# ===============================
@st.cache_resource
def load_index(mode):
    files = get_qa_files(mode)
    if not files:
        return None

    docs = []
    for f in files:
        docs.extend(TextLoader(f, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.from_documents(chunks, embeddings)
    return index

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
# SMART ANSWER (CHATGPT STYLE)
# ===============================
def smart_answer(question, docs, history):
    context = "\n\n".join(d.page_content for d in docs)[:2500]
    history_text = "\n".join(
        f"Q: {h['q']}\nA: {h['a']}"
        for h in history[-3:]
    )

    prompt = f"""
You are an internal assistant.
Answer clearly and concisely.
Do NOT list SOP sections.
Do NOT repeat the SOP text.
Answer like a knowledgeable colleague.

Conversation so far:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""
    return get_llm().invoke(prompt).content.strip()

# ===============================
# UI – QUESTION
# ===============================
st.subheader(f"{tool_mode} Assistant")

with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")
    col1, col2 = st.columns([1, 1])
    ask = col1.form_submit_button("Ask")
    clear = col2.form_submit_button("Clear chat")

# ===============================
# CLEAR CHAT
# ===============================
if clear:
    st.session_state.chat[tool_mode] = []
    st.rerun()

# ===============================
# HANDLE QUESTION
# ===============================
if ask and q.strip():
    index = load_index(tool_mode)
    if index is None:
        st.error("No Q&A files found.")
        st.stop()

    docs = index.similarity_search(q, k=4)

    if answer_mode == "Ultra-Fast":
        answer = docs[0].page_content.strip()
    else:
        answer = smart_answer(q, docs, st.session_state.chat[tool_mode])

    st.session_state.chat[tool_mode].append({
        "q": q,
        "a": answer
    })
    st.rerun()

# ===============================
# CHAT HISTORY (PER TOOL)
# ===============================
for item in st.session_state.chat[tool_mode]:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:10px;">
        <b>Q:</b> {item['q']}<br><br>
        <b>A:</b> {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )
