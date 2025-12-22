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
# SESSION STATE
# ===============================
st.session_state.setdefault("chat", [])
st.session_state.setdefault("topic", "")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Tool")
    tool_mode = st.radio(
        "",
        ["General", "Bayut", "Dubizzle"],
        index=0
    )

    st.markdown(" ")
    st.header("Answer mode")
    answer_mode = st.radio(
        "",
        ["Ultra-Fast", "Thinking"],
        index=0
    )

    st.markdown(" ")
    if st.button("Rebuild Index"):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        st.cache_resource.clear()
        st.success("Indexes cleared. They will rebuild automatically.")

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
# LLM (THINKING ONLY)
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
# INTENT DETECTION (THE FIX)
# ===============================
def is_tool_question(q: str) -> bool:
    q = q.lower().strip()
    triggers = [
        "what is this tool",
        "what is this app",
        "what are you",
        "who are you",
        "what can you do",
        "what is bayut & dubizzle ai content assistant",
        "what is bayut and dubizzle ai content assistant",
    ]
    return any(t in q for t in triggers)

def tool_description() -> str:
    return (
        "The Bayut & Dubizzle AI Content Assistant is an internal tool that helps "
        "content, marketing, and operations teams quickly find answers from "
        "approved SOPs, guidelines, and internal documentation.\n\n"
        "It supports Bayut, Dubizzle, or both, and can answer questions naturally "
        "without needing to search documents manually."
    )

# ===============================
# CONTEXT LOGIC
# ===============================
def clean_query(q: str) -> str:
    q = re.sub(r"\blunch\b", "launch", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcampains\b", "campaigns", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpm\b", "paid marketing", q, flags=re.IGNORECASE)
    return q.strip()

def update_topic(current: str, question: str) -> str:
    q = question.lower()
    if q.startswith(("what is", "what are", "define", "explain")):
        return question
    return current or question

def resolve_query(topic: str, question: str) -> str:
    return f"{topic}. Follow-up question: {question}" if topic else question

# ===============================
# ANSWERS
# ===============================
def ultra_fast_answer(q, docs):
    text = " ".join(d.page_content for d in docs)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    q_words = set(re.findall(r"\w+", q.lower()))
    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True
    )
    return " ".join(ranked[:2]).strip() or "I couldn’t find a clear answer."

def thinking_answer(q, docs):
    llm = get_llm()
    context = "\n\n".join(d.page_content for d in docs)[:2000]

    prompt = f"""
Answer clearly and directly.
Stay on topic.
Do NOT dump SOPs unless asked.

Context:
{context}

Question:
{q}

Answer:
"""
    return llm.invoke(prompt).content.strip()

# ===============================
# UI
# ===============================
st.subheader("Ask your internal question")

q = st.text_input("Question")

if st.button("Ask"):
    q_clean = clean_query(q)

    # 🔴 HARD STOP — TOOL QUESTIONS NEVER SEARCH
    if is_tool_question(q_clean):
        ans = tool_description()
        st.session_state.chat.append({"q": q_clean, "a": ans})
        st.stop()

    resolved = resolve_query(st.session_state.topic, q_clean)
    docs = index.similarity_search(resolved, k=3)

    if answer_mode == "Ultra-Fast":
        ans = ultra_fast_answer(q_clean, docs)
    else:
        ans = thinking_answer(q_clean, docs)

    st.session_state.topic = update_topic(st.session_state.topic, q_clean)
    st.session_state.chat.append({"q": q_clean, "a": ans})

# ===============================
# CHAT HISTORY
# ===============================
for item in st.session_state.chat:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:8px;">
          <b>{item['q']}</b><br>
          {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )
