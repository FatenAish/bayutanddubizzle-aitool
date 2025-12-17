import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Only used in Thinking mode
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container{
        max-width: 100% !important;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }

      /* Buttons */
      div[data-testid="stForm"] button{
        height: 42px;
        white-space: nowrap;
        padding-left: 14px;
        padding-right: 14px;
      }

      /* Make the 2 button columns almost touch */
      .tight-cols [data-testid="column"]{
        padding-left: 0.02rem !important;
        padding-right: 0.02rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# SESSION STATE (ONLY LATEST Q/A)
# =========================================
if "last_q" not in st.session_state:
    st.session_state.last_q = ""
if "last_a" not in st.session_state:
    st.session_state.last_a = ""


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])

    answer_mode = st.radio(
        "Answer mode",
        ["Ultra-Fast", "Thinking"],
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
        Fast internal assistant
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
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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
# TEXT CLEANING
# =========================================
STOPWORDS = {
    "what","is","are","the","a","an","and","or","to","of","in","on","for",
    "how","does","do","did","when","where","why","which","who","with","from",
    "define","defines","tell","me","about","please"
}

QUESTION_STARTERS = ("what", "how", "why", "which", "who", "when", "where")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def strip_q_labels(s: str) -> str:
    return re.sub(r"\bQ\d+\s*[-–:]\s*", "", s, flags=re.IGNORECASE).strip()

def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+\s*[\)\-–:]\s*", "", s).strip()

def strip_sop_prefix(s: str) -> str:
    s = normalize_spaces(s)
    m = re.search(r"\bSOP\b", s[:220])
    if m:
        s = s[m.end():].lstrip(" :–-")
        s = normalize_spaces(s)
    return s

def remove_embedded_questions(text: str) -> str:
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.endswith("?"):
            continue
        first_word = re.split(r"\s+", p.lower(), maxsplit=1)[0]
        if first_word in QUESTION_STARTERS:
            continue
        cleaned.append(p)
    return " ".join(cleaned).strip()

def postprocess_answer(answer: str, question: str) -> str:
    a = normalize_spaces(answer)
    a = strip_q_labels(a)
    a = strip_leading_numbering(a)
    a = strip_sop_prefix(a)

    q = normalize_spaces(question)
    if q and q.lower() in a.lower():
        a = re.sub(re.escape(q), "", a, flags=re.IGNORECASE).strip()

    a = remove_embedded_questions(a)
    return a.strip()

def split_sentences(text: str):
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# =========================================
# ULTRA-FAST (NO LLM)
# =========================================
def extractive_answer(question: str, docs) -> str:
    if not docs:
        return "No relevant internal content found."

    text = normalize_spaces(docs[0].page_content)

    q_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", question)]
    q_words = [w for w in q_words if w not in STOPWORDS and len(w) > 2]
    q_set = set(q_words)

    sents = split_sentences(text)
    if not sents:
        return postprocess_answer(text[:600], question)

    scored = []
    for s in sents:
        words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9']+", s))
        scored.append((len(words & q_set), s))

    scored.sort(key=lambda x: x[0], reverse=True)

    best = [s for score, s in scored[:5] if score > 0]
    if not best:
        best = sents[:3]

    answer = " ".join(best[:3])
    return postprocess_answer(answer[:900], question)


# =========================================
# THINKING (LLM)
# =========================================
@st.cache_resource
def get_llm():
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
Do not repeat the question.
Do not include any other questions.
Return ONLY the answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()

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

def thinking_answer(question: str, docs) -> str:
    if not docs:
        return "No relevant internal content found."

    context = "\n".join(normalize_spaces(d.page_content) for d in docs)[:1600]
    raw = cached_llm_answer(question, context)
    return postprocess_answer(raw, question)


# =========================================
# UI
# =========================================
if mode == "General":
    st.subheader("Ask your internal question")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Question", placeholder="Type your question and press Enter…")

        st.markdown("<div class='tight-cols'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([0.9, 1.35, 12.0])
        with c1:
            ask = st.form_submit_button("Ask")
        with c2:
            clear = st.form_submit_button("Clear chat")
        st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.last_q = ""
        st.session_state.last_a = ""
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif index is None:
            st.error("Index not available. Make sure there are .txt files inside /data.")
        else:
            docs = index.similarity_search(question, k=2)

            if answer_mode == "Ultra-Fast":
                answer = extractive_answer(question, docs)
            else:
                with st.spinner("Thinking..."):
                    answer = thinking_answer(question, docs)

            st.session_state.last_q = question.strip()
            st.session_state.last_a = answer.strip()
            st.rerun()

    if st.session_state.last_q and st.session_state.last_a:
        st.markdown(
            f"""
            <div style="
                background:#FFFFFF;
                padding:16px;
                border-radius:10px;
                border:1px solid #EEE;
                margin-top:14px;">
                <b>{st.session_state.last_q}</b><br><br>
                {st.session_state.last_a}
            </div>
            """,
            unsafe_allow_html=True
        )

elif mode == "Bayut":
    st.markdown("<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>", unsafe_allow_html=True)

elif mode == "Dubizzle":
    st.markdown("<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>", unsafe_allow_html=True)
