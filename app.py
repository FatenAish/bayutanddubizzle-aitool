import os
import re
import difflib

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


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

def _list_txt_files(path):
    if not os.path.isdir(path):
        return []
    return [
        name for name in os.listdir(path)
        if name.lower().endswith(".txt")
    ]


def _has_txt_files(path):
    return len(_list_txt_files(path)) > 0


def _find_txt_dir(root, max_depth=3):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if any(name.lower().endswith(".txt") for name in filenames):
            return dirpath
    return None


def _get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def _get_embeddings_client():
    api_key = _get_openai_api_key()
    if not api_key:
        return None
    return OpenAIEmbeddings(api_key=api_key)


def resolve_data_dir():
    env_dir = os.getenv("DATA_DIR")
    if env_dir and _has_txt_files(env_dir):
        return env_dir

    preferred = "/app/data"
    alt_root = "/data"
    local_dir = os.path.join(os.path.dirname(__file__), "data")
    cwd_dir = os.path.join(os.getcwd(), "data")
    candidates = [preferred, alt_root, local_dir, cwd_dir]
    for candidate in candidates:
        if _has_txt_files(candidate):
            return candidate

    search_roots = [os.path.dirname(__file__), os.getcwd()]
    for root in search_roots:
        found = _find_txt_dir(root)
        if found:
            return found

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return preferred


DATA_DIR = resolve_data_dir()
INDEX_PATH = os.path.join(DATA_DIR, "qa_faiss_index")

QUESTION_PATTERN = re.compile(r"^\s*Q\s*\d+\s*[\)\.\-:\u2013\u2014]\s*(.+)$", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^\s*[\u2022\u25cf]\s*")
MIN_MATCH_SCORE = 0.35


# =========================================
# FUNCTIONS
# =========================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def similarity_score(text_a, text_b):
    normalized_a = normalize_text(text_a)
    normalized_b = normalize_text(text_b)
    if not normalized_a or not normalized_b:
        return 0.0
    if normalized_a in normalized_b or normalized_b in normalized_a:
        return 1.0
    seq_score = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
    tokens_a = set(normalized_a.split())
    tokens_b = set(normalized_b.split())
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return max(seq_score, jaccard)


def format_answer(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        stripped = BULLET_PATTERN.sub("- ", stripped)
        lines.append(stripped)
    return "\n".join(lines).strip()


def parse_qa_pairs(text, source_name):
    pairs = []
    current_question = None
    current_answer_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = QUESTION_PATTERN.match(line)
        if match:
            if current_question:
                answer = format_answer("\n".join(current_answer_lines))
                if answer:
                    pairs.append(
                        {"question": current_question, "answer": answer, "source": source_name}
                    )
            current_question = match.group(1).strip()
            current_answer_lines = []
            continue

        if current_question:
            if line and set(line) == {"_"}:
                continue
            current_answer_lines.append(raw_line.rstrip())

    if current_question:
        answer = format_answer("\n".join(current_answer_lines))
        if answer:
            pairs.append({"question": current_question, "answer": answer, "source": source_name})
    return pairs


def load_qa_pairs():
    if not os.path.isdir(DATA_DIR):
        return []
    files = _list_txt_files(DATA_DIR)
    pairs = []
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as handle:
            pairs.extend(parse_qa_pairs(handle.read(), filename))
    return pairs


def build_faiss_index():
    """
    Build a FAISS vector index from Q/A pairs in .txt files.
    """
    embeddings = _get_embeddings_client()
    if embeddings is None:
        return False

    txt_files = _list_txt_files(DATA_DIR)
    if not txt_files:
        return False

    qa_pairs = load_qa_pairs()
    if qa_pairs:
        texts = [pair["question"] for pair in qa_pairs]
        metadatas = [
            {
                "question": pair["question"],
                "answer": pair["answer"],
                "source": pair["source"],
                "mode": "qa",
            }
            for pair in qa_pairs
        ]

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local(INDEX_PATH)
        return True

    documents = []
    for filename in txt_files:
        loader = TextLoader(os.path.join(DATA_DIR, filename))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["mode"] = "chunk"

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    return True


def load_index():
    """
    Load FAISS index if exists.
    """
    embeddings = _get_embeddings_client()
    if embeddings is None:
        return None
    if not os.path.exists(INDEX_PATH):
        return None
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def rag_query(question):
    """
    Retrieve a matching answer directly from the files.
    """
    if _get_openai_api_key() is None:
        return "Missing OPENAI_API_KEY. Add it in Streamlit secrets or environment variables."

    index = load_index()
    if index is None:
        return "Index missing. Please rebuild the index."

    results = index.similarity_search(question, k=5)
    if not results:
        return "I couldn't find a matching answer in the files."

    if results[0].metadata.get("mode") == "qa":
        best_doc = None
        best_score = 0.0
        for doc in results:
            candidate_question = doc.metadata.get("question", doc.page_content)
            score = similarity_score(question, candidate_question)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score < MIN_MATCH_SCORE:
            return "I couldn't find a matching answer in the files."

        answer = best_doc.metadata.get("answer")
        if not answer:
            return "I couldn't find a matching answer in the files."

        return answer

    return format_answer(results[0].page_content)


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
    📁 <b>DATA DIR:</b> {DATA_DIR} <br>
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
        if _get_openai_api_key() is None:
            st.error("Missing OPENAI_API_KEY. Add it in Streamlit secrets or environment variables.")
        else:
            with st.spinner("Rebuilding FAISS index..."):
                success = build_faiss_index()
                if success:
                    st.success("Index rebuilt successfully! Refresh page.")
                else:
                    st.error(f"No .txt files found in {DATA_DIR} folder.")

elif mode == "Bayut":
    st.markdown("<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>", unsafe_allow_html=True)

elif mode == "Dubizzle":
    st.markdown("<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>", unsafe_allow_html=True)
