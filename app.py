import os
import re
import unicodedata
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

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
# TOOL HEADING
# ===============================
if tool_mode == "Bayut":
    st.subheader("Ask Bayut Anything")
elif tool_mode == "Dubizzle":
    st.subheader("Ask Dubizzle Anything")
else:
    st.subheader("General Assistant")

# ===============================
# FILE HELPERS
# ===============================
def list_all_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")]

def get_qa_files(mode: str):
    files = []
    for fp in list_all_txt_files():
        f = os.path.basename(fp)
        lf = f.lower()
        if not lf.endswith("-qa.txt"):
            continue

        if mode == "Bayut" and not lf.startswith("bayut"):
            continue
        if mode == "Dubizzle" and not lf.startswith("dubizzle"):
            continue

        files.append(fp)
    return files

# ===============================
# CLEAN ANSWER (filecite + spacing + formatting)
# ===============================
def clean_answer(text: str) -> str:
    if not text:
        return ""

    # Normalize unicode (fixes invisible junk)
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width / directional chars
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", text)

    # Remove private-use + replacement chars (boxes)
    text = re.sub(r"[\uE000-\uF8FF\uFFFD]", "", text)

    # Remove exact marker
    text = re.sub(r"", " ", text)

    # Remove any spaced-out "filecite" (f i l e c i t e)
    text = re.sub(r"(?i)f\s*i\s*l\s*e\s*c\s*i\s*t\s*e", " ", text)

    # Remove any token containing filecite
    text = re.sub(r"(?i)\S*filecite\S*", " ", text)

    # Remove turn file patterns (turn3file0, turn 3 file 0, turn_3_file_0)
    text = re.sub(r"(?i)turn\s*\d+\s*file\s*\d+", " ", text)
    text = re.sub(r"(?i)turn\d+file\d+", " ", text)
    text = re.sub(r"(?i)\S*turn\d+\s*file\d+\S*", " ", text)

    # Collapse spaced letters sequences: "S u b - e d i t o r" -> "Sub-editor"
    # First normalize hyphen spacing
    text = re.sub(r"\s*-\s*", "-", text)

    # Collapse many single-letter tokens into words
    def _collapse(m):
        return m.group(0).replace(" ", "")
    text = re.sub(r"(?:(?<=\b)[A-Za-z]\s+){2,}[A-Za-z](?=\b)", _collapse, text)

    # Improve numbered steps formatting
    text = re.sub(r"(?<!\n)(\d+)\.\s*", r"\n\1. ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ===============================
# PARSE Q/A PAIRS (NO CHUNKING!)
# ===============================
QA_PATTERN = re.compile(r"(?:^|\n)Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:\s*|\Z)", re.S)

def build_qa_documents(mode: str):
    qa_files = get_qa_files(mode)
    if not qa_files:
        return []

    docs = []
    for fp in qa_files:
        raw_docs = TextLoader(fp, encoding="utf-8").load()
        full_text = "\n".join(d.page_content for d in raw_docs)

        for q, a in QA_PATTERN.findall(full_text):
            q_clean = q.strip()
            a_clean = a.strip()
            if not q_clean or not a_clean:
                continue

            # store Q and A as one doc (perfect retrieval)
            docs.append(
                Document(
                    page_content=f"Q: {q_clean}\nA: {a_clean}",
                    metadata={"q": q_clean, "source": os.path.basename(fp)}
                )
            )
    return docs

# ===============================
# BUILD INDEX (PER MODE)
# ===============================
@st.cache_resource
def load_index(mode: str):
    docs = build_qa_documents(mode)
    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ===============================
# ANSWER PICKING (STRICT)
# ===============================
def pick_best_answer(question: str, retrieved_docs):
    q_words = set(re.findall(r"\w+", question.lower()))
    best_score = -1
    best_ans = None

    for d in retrieved_docs:
        q_text = (d.metadata.get("q") or "").lower()
        # score overlap with the stored question text
        score = len(q_words & set(re.findall(r"\w+", q_text)))

        if score > best_score:
            best_score = score
            # extract answer portion
            content = d.page_content
            if "\nA:" in content:
                best_ans = content.split("\nA:", 1)[1].strip()
            else:
                best_ans = content

    return clean_answer(best_ans) if best_ans else "I couldn’t find a clear answer to that."

def thinking_answer(question: str, retrieved_docs, history):
    # lightweight: bias to previous question words too
    q_words = set(re.findall(r"\w+", question.lower()))
    prev_words = set()
    if history:
        prev_words = set(re.findall(r"\w+", history[-1]["q"].lower()))

    best_score = -1
    best_ans = None

    for d in retrieved_docs:
        q_text = (d.metadata.get("q") or "").lower()
        words = set(re.findall(r"\w+", q_text))
        score = len(q_words & words) + len(prev_words & words)

        if score > best_score:
            best_score = score
            content = d.page_content
            best_ans = content.split("\nA:", 1)[1].strip() if "\nA:" in content else content

    return clean_answer(best_ans) if best_ans else "I couldn’t find a clear answer to that."

# ===============================
# DOWNLOAD SOP (ONLY WHEN ASKED)
# ===============================
def is_download_sop_request(question: str) -> bool:
    q = question.lower().strip()
    return ("download" in q or "dl" in q or "get" in q) and ("sop" in q or "sops" in q)

def pick_sop_files(mode: str, question: str):
    # downloadable = any txt that is NOT -QA.txt, filtered by brand
    all_txt = list_all_txt_files()
    files = []
    for fp in all_txt:
        name = os.path.basename(fp).lower()
        if name.endswith("-qa.txt"):
            continue
        if mode == "Bayut" and not name.startswith("bayut"):
            continue
        if mode == "Dubizzle" and not name.startswith("dubizzle"):
            continue
        files.append(fp)

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
# Buttons tight on the LEFT, tiny gap, Enter submits Ask
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    # wider columns so "Clear chat" doesn't wrap
    b1, b2, _sp = st.columns([2, 3, 15], gap="small")
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

    retrieved = index.similarity_search(q, k=8)

    if answer_mode == "Ultra-Fast":
        answer = pick_best_answer(q, retrieved)
    else:
        answer = thinking_answer(q, retrieved, st.session_state.chat[tool_mode])

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY (NEWEST FIRST)
# Question ONLY in colored bubble
# ===============================
def bubble_css(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "Dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

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
            label_name = os.path.basename(fp).replace(".txt", "").replace("_", " ")
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
