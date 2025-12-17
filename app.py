import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


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
if "chat" not in st.session_state:
    st.session_state.chat = []


# =========================================
# TITLE
# =========================================
st.markdown(
    """
    <h1 style='text-align:center; font-weight:800;'>
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">Dubizzle</span>
        AI Content Assistant
    </h1>
    <p style='text-align:center; color:#666;'>
        Fast internal AI assistant powered by internal SOPs
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================
# PATHS
# =========================================
DATA_DIR = "data"
INDEX_PATH = "/tmp/faiss_index"


# =========================================
# EMBEDDINGS (FAST + CACHED)
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# LLM (FASTEST SAFE FREE MODEL)
# =========================================
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)


# =========================================
# INDEX (LOAD ONCE)
# =========================================
@st.cache_resource
def load_index():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

    if not os.path.exists(DATA_DIR):
        return None

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        return None

    documents = []
    for f in files:
        loader = TextLoader(os.path.join(DATA_DIR, f), encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    index = FAISS.from_documents(chunks, get_embeddings())
    index.save_local(INDEX_PATH)
    return index


index = load_index()


# =========================================
# SMART ANSWER (TYPE-SAFE)
# =========================================
def generate_answer(question, docs):
    llm = get_llm()
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Answer clearly and briefly.
Use only the context.
Explain in your own words.

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)

    # 🔒 SAFETY: normalize output
    if isinstance(result, list):
        return result[0]["generated_text"]
    return result


# =========================================
# MAIN UI
# =========================================
st.subheader("Ask your internal question")

with st.form("ask_form", clear_on_submit=True):

    question = st.text_input(
        "Question",
        placeholder="Type your question and press Enter…"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        ask = st.form_submit_button("Ask")

    with col3:
        clear = st.form_submit_button("Clear chat")

    if clear:
        st.session_state.chat = []
        st.experimental_rerun()

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif index is None:
            st.error("Index not available.")
        else:
            docs = index.similarity_search(question, k=3)
            answer = generate_answer(question, docs)

            st.session_state.chat.append({
                "q": question,
                "a": answer
            })

            st.experimental_rerun()


# =========================================
# CHAT HISTORY
# =========================================
for item in reversed(st.session_state.chat):
    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            padding:14px;
            border-radius:8px;
            border:1px solid #EEE;
            margin-top:12px;">
            <b>Q:</b> {item["q"]}<br><br>
            <b>A:</b> {item["a"]}
        </div>
        """,
        unsafe_allow_html=True
    )
