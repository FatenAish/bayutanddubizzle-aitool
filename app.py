import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================
# PAGE SETUP
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])

# Logo + heading
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

DATA_DIR = "/app/data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")


# =========================================
# FUNCTIONS
# =========================================

def build_faiss_index():
    """
    Build a FAISS vector index from all .txt files in /data.
    """
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        return False

    documents = []
    for f in files:
        loader = TextLoader(os.path.join(DATA_DIR, f))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_PATH)
    return True


def load_index():
    """
    Load FAISS index if exists.
    """
    if not os.path.exists(INDEX_PATH):
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def rag_query(question):
    """
    Perform RAG search + LLM answering.
    """
    index = load_index()
    if index is None:
        return "Index missing. Please rebuild the index."

    docs = index.similarity_search(question, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = PromptTemplate.from_template("""
    You are an internal knowledge assistant. Use ONLY the context below to answer.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """)

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": question})


# =========================================
# INDEX STATUS MESSAGE (CUSTOM UI)
# =========================================

index_exists = os.path.exists(INDEX_PATH)

st.markdown(f"""
<div style="
    background:#FFF4D1;
    padding:15px;
    border:1px solid #FFE19C;
    border-radius:8px;
    color:#444;
    font-size:16px;
    margin-bottom:15px;">
    📁 <b>DATA DIR:</b> /app/data <br>
    {'🟢 Index is ready.' if index_exists else '🟡 No index found — add .txt files and click Rebuild Index.'}
</div>
""", unsafe_allow_html=True)


# =========================================
# MAIN UI (Same for all modes)
# =========================================

if mode == "General":
    st.subheader("Ask your internal question")

    question = st.text_input("Question")

    if st.button("Ask"):
        if not index_exists:
            st.error("⚠ Please rebuild index first.")
        else:
            answer = rag_query(question)
            st.markdown(
                f"""
                <div style='background:#F7F7F7; padding:15px;
                border-radius:8px; border:1px solid #DDD; margin-top:15px;'>
                    <b>Answer:</b><br>{answer}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Rebuild index button
    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding FAISS index..."):
            success = build_faiss_index()
            if success:
                st.success("Index rebuilt successfully! Refresh page.")
            else:
                st.error("No .txt files found in /data folder.")

elif mode == "Bayut":
    st.markdown("<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>", unsafe_allow_html=True)

elif mode == "Dubizzle":
    st.markdown("<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>", unsafe_allow_html=True)
