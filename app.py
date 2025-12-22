import os
import re
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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

TOOLS = ["General", "Bayut", "Dubizzle"]

# ===============================
# SESSION STATE (SAFE INIT)
# ===============================
if "chat" not in st.session_state:
    st.session_state.chat = {t: [] for t in TOOLS}

if "topic" not in st.session_state:
    st.session_state.topic = {t: "" for t in TOOLS}

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Tool")
    tool_mode = st.radio("", TOOLS, index=0)

    # ensure keys always exist
    st.session_state.chat.setdefault(tool_mode, [])
    st.session_state.topic.setdefault(tool_mode, "")

    st.markdown(" ")
    st.header("Answer mode")
    answer_mode = st.radio("", ["Ultra-Fast", "Thinking"], index=0)

    st.markdown(" ")
    if st.button("🔁 Rebuild Index"):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        st.cache_resource.clear()
        st.success("Indexes rebuilt")

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
# EMBEDDINGS (LOCAL ONLY)
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
        chunk_size=900,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index

index = load_index(tool_mode)

# ===============================
# QUESTION INTENT DETECTION
# ===============================
def is_definition(q: str) -> bool:
    return q.lower().startswith(("what is", "what are", "define"))

def is_who(q: str) -> bool:
    q = q.lower()
    return q.startswith("who ") or "who works" in q or "responsible" in q

# ===============================
# QUERY CLEANING
# ===============================
def clean_query(q: str) -> str:
    q = re.sub(r"\blunch\b", "launch", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcampains\b", "campaigns", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpm\b", "paid marketing", q, flags=re.IGNORECASE)
    return q.strip()

# ===============================
# ANSWER STRATEGIES
# ===============================
def definition_answer(question, docs):
    if not docs:
        return "I couldn’t find a clear definition in the documentation."

    text = " ".join(d.page_content for d in docs)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for s in sentences:
        if "is" in s.lower() or "are" in s.lower():
            return s.strip()

    return sentences[0].strip()

def role_answer(question, docs):
    text = " ".join(d.page_content for d in docs).lower()

    roles = []
    for s in text.split("."):
        if any(k in s for k in [
            "responsible",
            "sub-editor",
            "content team",
            "writer",
            "editor",
            "poc"
        ]):
            roles.append(s.strip())

    if roles:
        return " ".join(roles[:2]).capitalize()

    return "This responsibility is handled by the content and sub-editing teams."

# ===============================
# UI — INPUT + BUTTONS
# ===============================
st.subheader("Ask your internal question")

q = st.text_input("Question", label_visibility="collapsed")

btn1, btn2 = st.columns(2)
with btn1:
    ask_clicked = st.button("Ask", use_container_width=True)
with btn2:
    clear_clicked = st.button("Clear chat", use_container_width=True)

if clear_clicked:
    st.session_state.chat[tool_mode] = []
    st.session_state.topic[tool_mode] = ""
    st.rerun()

if ask_clicked:
    q_clean = clean_query(q)

    topic = st.session_state.topic[tool_mode]
    search_q = f"{topic}. {q_clean}" if topic else q_clean

    docs = index.similarity_search(search_q, k=4)

    if is_who(q_clean):
        ans = role_answer(q_clean, docs)

    elif is_definition(q_clean):
        ans = definition_answer(q_clean, docs)
        st.session_state.topic[tool_mode] = q_clean

    else:
        ans = definition_answer(q_clean, docs)

    st.session_state.chat[tool_mode].append({
        "q": q_clean,
        "a": ans
    })

# ===============================
# CHAT HISTORY (PER TOOL)
# ===============================
for item in st.session_state.chat.get(tool_mode, []):
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:8px;">
          <b>{item['q']}</b><br>
          {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )
