import os
import re
import unicodedata
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

AKA_TOOL_NAMES = {"General": "General", "Bayut": "Bayut", "Dubizzle": "Dubizzle"}

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "Dubizzle": []
})

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Select tool")
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)

    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    <p style="text-align:center;color:#666;">Internal AI Assistant</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# TOOL HEADING (MAIN)
# ===============================
if tool_mode == "Bayut":
    st.subheader("Ask Bayut Anything")
elif tool_mode == "Dubizzle":
    st.subheader("Ask Dubizzle Anything")
else:
    st.subheader("General Assistant")

# ===============================
# QA FILE SELECTION
# ===============================
def get_qa_files(mode: str):
    if not os.path.isdir(DATA_DIR):
        return []

    files = []
    for f in os.listdir(DATA_DIR):
        if not f.endswith("-QA.txt"):
            continue

        lf = f.lower()
        if mode == "Bayut" and not lf.startswith("bayut"):
            continue
        if mode == "Dubizzle" and not lf.startswith("dubizzle"):
            continue

        files.append(os.path.join(DATA_DIR, f))
    return files

# ===============================
# BUILD INDEX (LOCAL)
# ===============================
@st.cache_resource
def load_index(mode: str):
    qa_files = get_qa_files(mode)
    if not qa_files:
        return None

    docs = []
    for f in qa_files:
        docs.extend(TextLoader(f, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# ===============================
# CLEAN + NORMALIZE ANSWER (FIXES filecite + weird spacing)
# ===============================
def clean_answer(text: str) -> str:
    if not text:
        return ""

    # Normalize unicode (fixes lots of invisible junk / spacing issues)
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width and directional characters that cause spaced letters
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", text)

    # Remove private-use + replacement chars (render as boxes)
    text = re.sub(r"[\uE000-\uF8FF\uFFFD]", "", text)

    # Remove exact marker format if present
    text = re.sub(r"", " ", text)

    # Remove any token containing filecite (handles weird glued strings like ...filecite��turn3file0)
    text = re.sub(r"(?i)\S*filecite\S*", " ", text)

    # Remove any token containing turnXfileY
    text = re.sub(r"(?i)\S*turn\d+file\d+\S*", " ", text)

    # Sometimes citations appear as bare "filecite turn3file0"
    text = re.sub(r"(?i)\bfilecite\b", " ", text)
    text = re.sub(r"(?i)\bturn\d+file\d+\b", " ", text)

    # Fix "O p e n W o r d P r e s s" style spacing if it happens:
    # collapse spaces between single letters when there are many in a row
    text = re.sub(r"(?:(?<=\b)[A-Za-z]\s+){3,}[A-Za-z]\b",
                  lambda m: m.group(0).replace(" ", ""),
                  text)

    # Add line breaks for numbered steps (1. 2. 3.)
    text = re.sub(r"(?<!\n)(\d+)\.\s*", r"\n\1. ", text)

    # Clean extra spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ===============================
# STRICT ANSWER EXTRACTION
# ===============================
def extract_answer_only(question, docs):
    q_words = set(re.findall(r"\w+", question.lower()))
    best_score = 0
    best_answer = None

    for d in docs:
        text = d.page_content
        blocks = re.split(r"\nQ:\s*", text)

        for block in blocks:
            if "\nA:" not in block:
                continue

            q_part, a_part = block.split("\nA:", 1)
            q_text = q_part.strip().lower()
            a_text = a_part.strip()

            a_text = re.split(r"\nQ:\s*", a_text)[0].strip()

            score = len(q_words & set(re.findall(r"\w+", q_text)))
            if score > best_score:
                best_score = score
                best_answer = a_text

    return clean_answer(best_answer) if best_answer else "I couldn’t find a clear answer to that."

# ===============================
# THINKING MODE (CHAT-AWARE, STILL STRICT)
# ===============================
def thinking_answer(question, docs, history):
    q_words = set(re.findall(r"\w+", question.lower()))
    prev_words = set()

    if history:
        prev_words = set(re.findall(r"\w+", history[-1]["q"].lower()))

    best_score = 0
    best_answer = None

    for d in docs:
        text = d.page_content
        blocks = re.split(r"\nQ:\s*", text)

        for block in blocks:
            if "\nA:" not in block:
                continue

            q_part, a_part = block.split("\nA:", 1)
            q_text = q_part.lower()
            a_text = re.split(r"\nQ:\s*", a_part)[0].strip()

            score = len(q_words & set(re.findall(r"\w+", q_text)))
            score += len(prev_words & set(re.findall(r"\w+", q_text)))

            if score > best_score:
                best_score = score
                best_answer = a_text

    return clean_answer(best_answer) if best_answer else "I couldn’t find a clear answer to that."

# ===============================
# DOWNLOAD SOP INTENT
# ===============================
def is_download_sop_request(question: str) -> bool:
    q = question.lower().strip()
    return ("download" in q or "dl" in q or "get" in q) and ("sop" in q or "sops" in q)

def pick_sop_files(mode: str, question: str):
    files = get_qa_files(mode)
    if not files:
        return []

    stop = {"download", "dl", "get", "sop", "sops", "please", "the", "a", "an", "for", "to", "of"}
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", question.lower()) if t not in stop]

    if not tokens:
        return files

    scored = []
    for fp in files:
        name = os.path.basename(fp).lower()
        score = sum(1 for t in tokens if t in name)
        scored.append((score, fp))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    if best_score == 0:
        return files
    return [fp for s, fp in scored if s == best_score]

# ===============================
# UI – QUESTION
# Ask + Clear beside each other on the LEFT (tiny space)
# Enter submits Ask ✅
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")
    b1, b2, _sp = st.columns([1, 1, 10], gap="small")
    ask = b1.form_submit_button("Ask")
    clear = b2.form_submit_button("Clear chat")

# ===============================
# CLEAR CHAT
# ===============================
if clear:
    st.session_state.chat[tool_mode] = []
    st.rerun()

# ===============================
# HANDLE QUESTION
# ===============================
if ask and q.strip():
    if is_download_sop_request(q):
        files = pick_sop_files(tool_mode, q)
        if not files:
            st.session_state.chat[tool_mode].append({"q": q, "a": "No SOP files found to download.", "downloads": []})
        else:
            st.session_state.chat[tool_mode].append({"q": q, "a": "Here are the SOP files you can download:", "downloads": files})
        st.rerun()

    index = load_index(tool_mode)
    if index is None:
        st.error("No Q&A files found.")
        st.stop()

    docs = index.similarity_search(q, k=5)

    if answer_mode == "Ultra-Fast":
        answer = extract_answer_only(q, docs)
    else:
        answer = thinking_answer(q, docs, st.session_state.chat[tool_mode])

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY (NEWEST FIRST)
# - Question ONLY in colored bubble
# ===============================
def bubble_css(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"  # light green
    if mode == "Dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"  # light red
    return "background:#F5F6F8;border:1px solid #E2E5EA;"      # neutral

style = bubble_css(tool_mode)
chat_list = st.session_state.chat[tool_mode]

for i in range(len(chat_list) - 1, -1, -1):
    item = chat_list[i]

    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:8px;">
            {item['q']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(item.get("a", ""))

    downloads = item.get("downloads", [])
    if downloads:
        cols = st.columns(min(3, len(downloads)))
        for j, fp in enumerate(downloads):
            col = cols[j % len(cols)]
            label_name = os.path.basename(fp).replace("-QA.txt", "").replace("_", " ")
            with col:
                with open(fp, "rb") as f:
                    st.download_button(
                        label=f"Download {label_name}",
                        data=f.read(),
                        file_name=os.path.basename(fp),
                        mime="text/plain",
                        key=f"dl_{tool_mode}_{i}_{j}_{os.path.basename(fp)}"
                    )

    st.markdown("---")
