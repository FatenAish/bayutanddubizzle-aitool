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

# ✅ LEFT-ALIGNED LAYOUT (NOT CENTER)
st.markdown(
    """
    <style>
      /* Full width + left aligned */
      .block-container {
        max-width: 100% !important;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }

      /* Buttons: same height + no wrap */
      div[data-testid="stForm"] button {
        width: 100%;
        height: 42px;
        white-space: nowrap;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# SESSION STATE
# =========================================
if "chat" not in st.session_state:
    st.session_state.chat = []


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])


# =========================================
# TITLE (keep centered if you like)
# =========================================
st.markdown(
    """
    <h1 style='text-align:center; font-weight:800; margin-bottom:6px;'>
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">Dubizzle</span>
        AI Content Assistant
    </h1>
    <p style='text-align:center; color:#666; margin-top:0;'>
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
# EMBEDDINGS (CACHED)
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# LLM (FASTER)
# =========================================
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=128,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)


# =========================================
# INDEX (LOAD OR BUILD ONCE)
# =========================================
@st.cache_resource
def load_or_build_index():
    embeddings = get_embeddings()

    # Load if exists
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Build if missing
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
        chunk_size=650,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index


index = load_or_build_index()


# =========================================
# ANSWER CACHE (repeat Q = instant)
# =========================================
@st.cache_data(show_spinner=False)
def cached_llm_answer(question: str, context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are an internal Bayut & Dubizzle assistant.

Rules:
- Answer clearly and professionally.
- Use ONLY the context.
- Do NOT copy SOP text or list Q1/Q2/Q3.
- If the context doesn't contain the answer, say: "I couldn't find this in the provided SOPs."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # LangChain version-safe calling
    try:
        result = llm.invoke(prompt)
    except Exception:
        result = llm(prompt) if callable(llm) else str(llm)

    # Normalize output types
    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return str(result[0]).strip()

    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"].strip()

    return str(result).strip()


def generate_answer(question, docs):
    context = "\n".join(d.page_content for d in docs)
    context = context[:2500]  # cap context for speed
    return cached_llm_answer(question, context)


# =========================================
# UI
# =========================================
if mode == "General":
    st.subheader("Ask your internal question")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Question", placeholder="Type your question and press Enter…")

        # Buttons next to each other and centered under the input
        left, b1, b2, right = st.columns([5, 2, 2, 5])

        with b1:
            ask = st.form_submit_button("Ask")

        with b2:
            clear = st.form_submit_button("Clear chat")

    if clear:
        st.session_state.chat = []
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif index is None:
            st.error("Index not available. Make sure there are .txt files inside /data.")
        else:
            with st.spinner("Thinking..."):
                docs = index.similarity_search(question, k=2)  # fewer docs = faster
                answer = generate_answer(question, docs)

            st.session_state.chat.append({"q": question, "a": answer})
            st.rerun()

    # Chat history
    for item in reversed(st.session_state.chat):
        st.markdown(
            f"""
            <div style="
                background:#FFFFFF;
                padding:14px;
                border-radius:10px;
                border:1px solid #EEE;
                margin-top:12px;">
                <b>Q:</b> {item["q"]}<br><br>
                <b>A:</b> {item["a"]}
            </div>
            """,
            unsafe_allow_html=True
        )

elif mode == "Bayut":
    st.markdown("<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>", unsafe_allow_html=True)

elif mode == "Dubizzle":
    st.markdown("<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>", unsafe_allow_html=True)
