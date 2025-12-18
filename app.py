# =========================================
# ENV
# =========================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Thinking mode
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

      div[data-testid="stForm"] button{
        height: 42px;
        white-space: nowrap;
        padding-left: 14px;
        padding-right: 14px;
      }

      .tight-cols [data-testid="column"]{
        padding-left: 0.02rem !important;
        padding-right: 0.02rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================
# SESSION STATE
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
# PATHS PER MODULE
# =========================================
def get_paths(selected_mode):
    base_index = "/tmp/faiss_index"

    if selected_mode == "Bayut":
        return "data/bayut", f"{base_index}_bayut"
    elif selected_mode == "Dubizzle":
        return "data/dubizzle", f"{base_index}_dubizzle"
    else:
        return "data/general", f"{base_index}_general"


DATA_DIR, INDEX_PATH = get_paths(mode)


# =========================================
# EMBEDDINGS
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================
# INDEX LOADER
# =========================================
@st.cache_resource
def load_or_build_index(data_dir, index_path):
    embeddings = get_embeddings()

    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    if not os.path.exists(data_dir):
        return None

    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not files:
        return None

    documents = []
    for f in files:
        loader = TextLoader(
            os.path.join(data_dir, f),
            encoding="utf-8"
        )
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(documents)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index


index = load_or_build_index(DATA_DIR, INDEX_PATH)


# =========================================
# TEXT CLEANING
# =========================================
STOPWORDS = {
    "what","is","are","the","a","an","and","or","to","of","in","on","for",
    "how","does","do","did","when","where","why","which","who","with","from",
    "define","defines","tell","me","about","please"
}

QUESTION_STARTERS = ("what", "how", "why", "which", "who", "when", "where")

def normalize_spaces(s):
    return re.sub(r"\s+", " ", s).strip()

def remove_questions(text):
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    clean = []
    for p in parts:
        if p.endswith("?"):
            continue
        if p.lower().startswith(QUESTION_STARTERS):
            continue
        clean.append(p)
    return " ".join(clean)

def postprocess_answer(answer, question):
    a = normalize_spaces(answer)
    q = normalize_spaces(question)

    if q.lower() in a.lower():
        a = re.sub(re.escape(q), "", a, flags=re.IGNORECASE)

    a = remove_questions(a)
    return a.strip()


# =========================================
# ULTRA FAST (NO LLM)
# =========================================
def extractive_answer(question, docs):
    if not docs:
        return "No relevant internal content found."

    text = normalize_spaces(docs[0].page_content)
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)

    q_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9']+", question)
        if w.lower() not in STOPWORDS and len(w) > 2
    }

    scored = []
    for s in sentences:
        words = set(re.findall(r"[a-zA-Z0-9']+", s.lower()))
        scored.append((len(words & q_words), s))

    scored.sort(reverse=True)
    best = [s for score, s in scored[:3] if score > 0]

    answer = " ".join(best) if best else sentences[0]
    return postprocess_answer(answer[:900], question)


# =========================================
# THINKING MODE (LLM)
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
def cached_llm_answer(question, context):
    llm = get_llm()

    prompt = f"""
Answer clearly and briefly using ONLY the context.
Do not repeat the question.
Do not ask questions.
Return ONLY the answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return llm.invoke(prompt).strip()

def thinking_answer(question, docs):
    if not docs:
        return "No relevant internal content found."

    context = "\n".join(d.page_content for d in docs)[:1600]
    raw = cached_llm_answer(question, context)
    return postprocess_answer(raw, question)


# =========================================
# MAIN UI (ONE UI FOR ALL MODULES)
# =========================================
st.subheader(f"Ask your internal question ({mode})")

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input(
        "Question",
        placeholder=f"Ask a {mode} question…"
    )

    st.markdown("<div class='tight-cols'>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([0.9, 1.35, 12.0])
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
        st.error(f"No index found for {mode}. Check /data/{mode.lower()}")
    else:
        docs = index.similarity_search(question, k=2)

        if answer_mode == "Ultra-Fast":
            answer = extractive_answer(question, docs)
        else:
            with st.spinner("Thinking..."):
                answer = thinking_answer(question, docs)

        st.session_state.last_q = question
        st.session_state.last_a = answer
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
