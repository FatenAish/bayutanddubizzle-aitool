import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Only loaded if you pick "Smart (LLM)" mode
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# LEFT aligned layout + buttons style
st.markdown(
    """
    <style>
      .block-container{
        max-width: 100% !important;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }
      div[data-testid="stForm"] button{
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

    answer_mode = st.radio(
        "Answer mode",
        ["Ultra-fast (no LLM)", "Smart (LLM - slower)"],
        index=0
    )

# =========================================
# TITLE
# =========================================
st.markdown(
    """
    <h1 style='text-align:center; font-weight:800; margin-bottom:6px;'>
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">Dubizzle</span>
        AI Content Assistant
    </h1>
    <p style='text-align:center; color:#666; margin-top:0;'>
        Fast internal assistant powered by internal SOPs
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
# INDEX (LOAD OR BUILD ONCE)
# =========================================
@st.cache_resource
def load_or_build_index():
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_PATH)
    return index


index = load_or_build_index()


# =========================================
# ULTRA-FAST ANSWER (NO LLM)
# =========================================
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "how", "does", "do", "did", "when", "where", "why", "which", "who", "with", "from",
    "define", "defines", "tell", "me", "about", "please"
}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # remove common SOP Q labels like "Q1 –", "Q2 -"
    s = re.sub(r"\bQ\d+\s*[-–:]\s*", "", s, flags=re.IGNORECASE)
    return s

def split_sentences(text: str):
    # simple sentence splitter
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def extractive_answer(question: str, docs) -> str:
    if not docs:
        return "No relevant internal content found."

    # Use only the top doc chunk (fast)
    text = clean_text(docs[0].page_content)

    # Score sentences by overlap with question keywords
    q_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", question)]
    q_words = [w for w in q_words if w not in STOPWORDS and len(w) > 2]
    q_set = set(q_words)

    sents = split_sentences(text)
    if not sents:
        return text[:400]

    scored = []
    for s in sents:
        words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9']+", s))
        score = len(words & q_set)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for score, s in scored[:3] if score > 0]  # top 3 relevant
    if not best:
        # fallback: first 2 sentences
        best = sents[:2]

    answer = " ".join(best)
    return answer[:900]  # keep it short & fast


# =========================================
# SMART (LLM) — OPTIONAL
# =========================================
@st.cache_resource
def get_llm():
    # smaller + fewer tokens = faster (still slower than no-LLM)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=96,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_data(show_spinner=False)
def cached_llm_answer(question: str, context: str) -> str:
    llm = get_llm()
    prompt = f"""
Answer clearly and briefly using ONLY the context.
Do not copy SOP Q1/Q2/Q3 text.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # version-safe calling
    try:
        result = llm.invoke(prompt)
    except Exception:
        result = llm(prompt) if callable(llm) else str(llm)

    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return str(result[0]).strip()

    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"].strip()

    return str(result).strip()

def smart_answer(question: str, docs) -> str:
    if not docs:
        return "No relevant internal content found."
    context = "\n".join(clean_text(d.page_content) for d in docs)
    context = context[:1800]  # cap for speed
    return cached_llm_answer(question, context)


# =========================================
# UI
# =========================================
if mode == "General":
    st.subheader("Ask your internal question")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Question", placeholder="Type your question and press Enter…")

        # ✅ Buttons beside each other on the LEFT
        b1, b2, spacer = st.columns([1.2, 1.6, 8])

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
            # 🔥 fewer docs = faster
            docs = index.similarity_search(question, k=2)

            if answer_mode == "Ultra-fast (no LLM)":
                answer = extractive_answer(question, docs)
            else:
                with st.spinner("Thinking..."):
                    answer = smart_answer(question, docs)

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
