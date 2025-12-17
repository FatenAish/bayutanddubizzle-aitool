import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


# =========================================
# PAGE SETUP
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])

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
        Smart internal knowledge assistant powered by internal SOPs and documents
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
# EMBEDDINGS (FREE)
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# LLM (FREE, SMART Q&A)
# =========================================
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)


# =========================================
# FAISS INDEX
# =========================================
def build_faiss_index():
    if not os.path.exists(DATA_DIR):
        return False

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        return False

    documents = []
    for file in files:
        loader = TextLoader(
            os.path.join(DATA_DIR, file),
            encoding="utf-8"
        )
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)

    return True


def load_index():
    if not os.path.exists(INDEX_PATH):
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def search_docs(question):
    index = load_index()
    if index is None:
        return []
    return index.similarity_search(question, k=5)


# =========================================
# SMART ANSWER GENERATION
# =========================================
def generate_answer(question, docs):
    llm = get_llm()

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an internal Bayut & Dubizzle AI assistant.

Your task:
- Answer the question clearly and professionally
- Explain in your own words
- Do NOT list SOP questions or sections
- Do NOT copy text verbatim
- Use the context ONLY

Context:
{context}

Question:
{question}

Answer:
"""

    return llm(prompt)


# =========================================
# AUTO BUILD INDEX (SILENT)
# =========================================
if not os.path.exists(INDEX_PATH):
    build_faiss_index()


# =========================================
# MAIN UI
# =========================================
if mode == "General":
    st.subheader("Ask your internal question")

    question = st.text_input("Question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            docs = search_docs(question)

            if not docs:
                st.info("No relevant internal content found.")
            else:
                answer = generate_answer(question, docs)

                st.markdown(
                    f"""
                    <div style="
                        background:#F7F7F7;
                        padding:18px;
                        border-radius:8px;
                        border:1px solid #DDD;
                        margin-top:15px;
                        font-size:16px;
                        line-height:1.6;">
                        {answer}
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
