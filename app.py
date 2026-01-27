import os
import re
import difflib

import streamlit as st

from langchain_community.vectorstores import FAISS
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

DATA_DIR = "/app/data"
INDEX_PATH = os.path.join(DATA_DIR, "qa_faiss_index")

QUESTION_PATTERN = re.compile(r"^\s*Q\s*\d+\s*[-\u2013:]\s*(.+)$", re.IGNORECASE)
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
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
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
    qa_pairs = load_qa_pairs()
    if not qa_pairs:
        return False

    texts = [pair["question"] for pair in qa_pairs]
    metadatas = [
        {"question": pair["question"], "answer": pair["answer"], "source": pair["source"]}
        for pair in qa_pairs
    ]

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
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
    Retrieve a matching answer directly from the files.
    """
    index = load_index()
    if index is None:
        return "Index missing. Please rebuild the index."

    results = index.similarity_search(question, k=5)
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question")
    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Please enter a question.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if not index_exists:
                response = "Please rebuild index first."
            else:
                response = rag_query(prompt)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Rebuild index button
    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding FAISS index..."):
            success = build_faiss_index()
            if success:
                st.success("Index rebuilt successfully! Refresh page.")
            else:
                st.error("No Q&A pairs found in .txt files.")

elif mode == "Bayut":
    st.markdown("<h2 style='color:#0E8A6D;'>Bayut Module (coming soon)</h2>", unsafe_allow_html=True)

elif mode == "Dubizzle":
    st.markdown("<h2 style='color:#D71920;'>Dubizzle Module (coming soon)</h2>", unsafe_allow_html=True)
