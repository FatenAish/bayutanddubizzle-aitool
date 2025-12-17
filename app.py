import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


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
        Fast internal knowledge search powered by internal content (.txt files in /data)
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================
# PATHS (CLOUD SAFE)
# =========================================
DATA_DIR = "data"                 # read-only
INDEX_PATH = "/tmp/faiss_index"   # writable


# =========================================
# EMBEDDINGS (FREE)
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# FAISS FUNCTIONS
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
                st.info("No relevant content found.")
            else:
                context = "\n\n".join(d.page_content for d in docs)
                st.markdown(
                    f"""
                    <div style="
                        background:#F7F7F7;
                        padding:15px;
                        border-radius:8px;
                        border:1px solid #DDD;
                        margin-top:15px;">
                        <b>Relevant content:</b><br><br>
                        {context}
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
