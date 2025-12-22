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
# SESSION STATE (CHATGPT-STYLE MEMORY)
# ===============================
st.session_state.setdefault("chat", [])     # full chat
st.session_state.setdefault("topic", "")    # running topic

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Answer mode")
    answer_mode = st.radio(
        "",
        ["Ultra-Fast", "Thinking"],
        index=0
    )

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
# EMBEDDINGS (LOCAL, FAST, STABLE)
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===============================
# LLM (REASONING ONLY)
# ===============================
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# ===============================
# LOAD / BUILD INDEX (ONCE)
# ===============================
@st.cache_resource
def load_index():
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index

index = load_index()

# ===============================
# HELPERS
# ===============================
def clean_query(q: str) -> str:
    q = re.sub(r"\blunch\b", "launch", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcampains\b", "campaigns", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpm\b", "paid marketing", q, flags=re.IGNORECASE)
    return q.strip()

def is_greeting(q: str):
    return q.lower() in {"hi", "hello", "hey", "مرحبا", "اهلا"}

def is_app_question(q: str):
    return "what is this app" in q.lower() or "what does this app do" in q.lower()

# ===============================
# 🔑 CHATGPT-STYLE TOPIC MANAGER (THE FIX)
# ===============================
def update_topic(current_topic: str, question: str) -> str:
    q = question.lower().strip()

    # new topic introducers → reset topic
    if q.startswith(("what is", "what are", "define", "explain")):
        return question

    # otherwise keep topic
    return current_topic or question

def resolve_query(topic: str, question: str) -> str:
    if topic:
        return f"{topic}. Follow-up question: {question}"
    return question

# ===============================
# ULTRA-FAST (NO LLM)
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

# ===============================
# THINKING (REASONING, NOT CHATGPT TONE)
# ===============================
def thinking_answer(q, docs):
    llm = get_llm()

    context = "\n\n".join(d.page_content for d in docs)[:2000]

    prompt = f"""
Answer clearly and directly.
Stay on the current topic.
Explain first; expand only if needed.
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

    if not q_clean:
        st.warning("Enter a question.")
        st.stop()

    if is_greeting(q_clean):
        ans = "Hello 👋 How can I help you?"
    elif is_app_question(q_clean):
        ans = (
            "This is an internal assistant for Bayut & Dubizzle. "
            "It answers questions based on internal content and SOPs."
        )
    else:
        # 🔑 CHATGPT-STYLE CONTEXT RESOLUTION
        resolved_query = resolve_query(
            st.session_state.topic,
            q_clean
        )

        docs = index.similarity_search(resolved_query, k=3)

        if answer_mode == "Ultra-Fast":
            ans = ultra_fast_answer(q_clean, docs)
        else:
            ans = thinking_answer(q_clean, docs)

        # update topic AFTER answering
        st.session_state.topic = update_topic(
            st.session_state.topic,
            q_clean
        )

    st.session_state.chat.append({"q": q_clean, "a": ans})

# ===============================
# DISPLAY CHAT HISTORY
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
