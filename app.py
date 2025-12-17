import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# =========================================
# PAGE CONFIG (FAST)
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =========================================
# SESSION STATE (CHAT MEMORY)
# =========================================
if "chat" not in st.session_state:
    st.session_state.chat = []


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("Options")

    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.experimental_rerun()

    mode = st.radio("Module", ["General", "Bayut", "Dubizzle"])


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
        Fast internal AI assistant powered by your SOPs
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
# EMBEDDINGS (CACHED – FAST)
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# LLM (FASTEST FREE OPTION)
# =========================================
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,      # 🔥 smaller = faster
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)


# =========================================
# BUILD INDEX (ONE TIME)
# =========================================
@st.cache_resource
def build_or_load_index():
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
    for file in files:
        loader = TextLoader(
            os.path.join(DATA_DIR, file),
            encoding="utf-8"
        )
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,      # 🔥 optimized
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    index = FAISS.from_documents(chunks, get_embeddings())
    index.save_local(INDEX_PATH)
    return index


# =========================================
# SMART ANSWER (FAST PROMPT)
# =========================================
def generate_answer(question, docs):
    llm = get_llm()

    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question clearly and briefly.
Use the context only.
Explain in your own words.

Context:
{context}

Question:
{question}

Answer:
"""
    return llm(prompt)


# =========================================
# LOAD INDEX ONCE
# =========================================
index = build_or_load_index()


# =========================================
# MAIN UI
# =========================================
if mode == "General":
    st.subheader("Ask your internal question")

    question = st.text_input("Question", key="question_input")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif index is None:
            st.error("Index not available.")
        else:
            docs = index.similarity_search(question, k=3)  # 🔥 fewer docs = faster
            answer = generate_answer(question, docs)

            st.session_state.chat.append(
                {"q": question, "a": answer}
            )

            st.experimental_rerun()

    # =====================================
    # CHAT HISTORY
    # =====================================
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

elif mode == "Bayut":
    st.markdown(
        "<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>",
        unsafe_allow_html=True
    )

elif mode == "Dubizzle":
    st.markdown(
        "<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>",
        unsafe_allow_html=True
    )
