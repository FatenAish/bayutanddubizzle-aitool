import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("chat", {
    "General": [],
    "Bayut": [],
    "dubizzle": []
})

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("Select tool")
    tool_mode = st.radio("", ["General", "Bayut", "dubizzle"], index=0)

    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">dubizzle</span>
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
elif tool_mode == "dubizzle":
    st.subheader("Ask dubizzle Anything")
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
        if mode == "dubizzle" and not lf.startswith("dubizzle"):
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
# CLEAN ANSWER (REMOVE FILECITE)
# ===============================
def clean_answer(text: str) -> str:
    if not text:
        return ""

    # Remove the exact marker
    text = re.sub(r"", "", text)

    # Extra safety: remove any leftover "filecite turnXfileY" fragments
    text = re.sub(r"filecite\s*turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)

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

            # Remove anything after next Q:
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

    # Optional filtering by keywords in the question (after removing common words)
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
        return files  # no match → show all for that mode
    return [fp for s, fp in scored if s == best_score]

# ===============================
# UI – QUESTION
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    # Buttons beside each other on the LEFT under the box
    b1, b2, _sp = st.columns([1, 1, 6])
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
    # If user requests SOP download in chat → show download buttons
    if is_download_sop_request(q):
        files = pick_sop_files(tool_mode, q)

        if not files:
            answer = "No SOP files found to download."
            st.session_state.chat[tool_mode].append({"q": q, "a": answer, "downloads": []})
        else:
            answer = "Here are the SOP files you can download:"
            st.session_state.chat[tool_mode].append({"q": q, "a": answer, "downloads": files})

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

    st.session_state.chat[tool_mode].append({
        "q": q,
        "a": answer
    })
    st.rerun()

# ===============================
# CHAT HISTORY (PER TOOL)
# ===============================
def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"  # light green
    if mode == "dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"  # light red
    return "background:#F5F6F8;border:1px solid #E2E5EA;"      # neutral

style = bubble_style(tool_mode)

for idx, item in enumerate(st.session_state.chat[tool_mode]):
    # Bubble
    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:10px;">
            <b>Q:</b> {item['q']}<br><br>
            {item['a']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Download buttons inside chat (only when requested)
    downloads = item.get("downloads", [])
    if downloads:
        cols = st.columns(min(3, len(downloads)))
        for i, fp in enumerate(downloads):
            col = cols[i % len(cols)]
            filename = os.path.basename(fp).replace("-QA.txt", "").replace("_", " ")

            with col:
                with open(fp, "rb") as f:
                    st.download_button(
                        label=f"Download {filename}",
                        data=f.read(),
                        file_name=os.path.basename(fp),
                        mime="text/plain",
                        key=f"dl_{tool_mode}_{idx}_{i}_{os.path.basename(fp)}"
                    )
