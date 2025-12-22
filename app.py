import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "dubizzle": []
})

st.session_state.setdefault("tool_mode", "General")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### Answer mode")
    answer_mode = st.radio("", ["Ultra-Fast", "Thinking"], index=0)

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
    <p style="text-align:center;color:#666;">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# CENTERED TOOL BUTTONS
# ===============================
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 2])

with c2:
    if st.button("Bayut AI Assistant", use_container_width=True):
        st.session_state.tool_mode = "Bayut"

with c3:
    if st.button("dubizzle AI Assistant", use_container_width=True):
        st.session_state.tool_mode = "dubizzle"

tool_mode = st.session_state.tool_mode

# ===============================
# QA FILE SELECTION
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
# BUILD INDEX
# ===============================
@st.cache_resource
def load_index(mode):
    qa_files = get_qa_files(mode)
    if not qa_files:
        return None

    docs = []
    for f in qa_files:
        docs.extend(TextLoader(f, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# ===============================
# CLEAN ANSWER (REMOVE CITATIONS)
# ===============================
def clean_answer(text):
    text = re.sub(r"", "", text)
    return text.strip()

# ===============================
# STRICT ANSWER EXTRACTION
# ===============================
def extract_answer_only(question, docs):
    q_words = set(re.findall(r"\w+", question.lower()))
    best_score = 0
    best_answer = None

    for d in docs:
        blocks = re.split(r"\nQ:\s*", d.page_content)
        for block in blocks:
            if "\nA:" not in block:
                continue

            q_part, a_part = block.split("\nA:", 1)
            q_text = q_part.lower()
            a_text = re.split(r"\nQ:\s*", a_part)[0].strip()

            score = len(q_words & set(re.findall(r"\w+", q_text)))
            if score > best_score:
                best_score = score
                best_answer = a_text

    return clean_answer(best_answer) if best_answer else "I couldn’t find a clear answer to that."

# ===============================
# THINKING MODE
# ===============================
def thinking_answer(question, docs, history):
    q_words = set(re.findall(r"\w+", question.lower()))
    prev_words = set()

    if history:
        prev_words = set(re.findall(r"\w+", history[-1]["q"].lower()))

    best_score = 0
    best_answer = None

    for d in docs:
        blocks = re.split(r"\nQ:\s*", d.page_content)
        for block in blocks:
            if "\nA:" not in block:
                continue

            q_part, a_part = block.split("\nA:", 1)
            q_text = q_part.lower()
            a_text = re.split(r"\nQ:\s*", a_part)[0].strip()

            score = len(q_words & set(re.findall(r"\w+", q_text)))
            score += len(prev_words & set(re.findall(r"\w+", q_text)))

            if score > best_score:
                best_score = score
                best_answer = a_text

    return clean_answer(best_answer) if best_answer else "I couldn’t find a clear answer to that."

# ===============================
# QUESTION INPUT
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

    docs = index.similarity_search(q, k=5)

    if answer_mode == "Ultra-Fast":
        answer = extract_answer_only(q, docs)
    else:
        answer = thinking_answer(q, docs, st.session_state.chat[tool_mode])

    st.session_state.chat[tool_mode].append({
        "q": q,
        "a": answer
    })

    st.rerun()

# ===============================
# CHAT HISTORY
# ===============================
for item in st.session_state.chat[tool_mode]:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:10px;">
        <b>Q:</b> {item['q']}<br><br>
        {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )
