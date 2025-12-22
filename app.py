import os
import re
import html
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

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
# SMALL HELPERS
# ===============================
def safe_columns(spec, gap="small"):
    """Backward compatible columns gap."""
    try:
        return st.columns(spec, gap=gap)
    except TypeError:
        return st.columns(spec)

def norm_mode(mode: str) -> str:
    return mode.strip()

def clean_answer(text: str) -> str:
    """Remove internal citation markers and leftovers."""
    if not text:
        return ""

    # Remove private-use citation blocks like: 
    text = re.sub(r"", "", text, flags=re.DOTALL)
    text = re.sub(r"", "", text, flags=re.DOTALL)

    # Remove any leftover "filecite turnXfileY" fragments
    text = re.sub(r"filecite\s*turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)

    # Trim extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def is_download_sop_request(question: str) -> bool:
    q = question.lower().strip()
    return (("download" in q) or ("dl" in q) or ("get" in q)) and (("sop" in q) or ("sops" in q))

# ===============================
# FILE SELECTION
# ===============================
def get_qa_files(mode: str):
    """Q&A retrieval source (only -QA.txt)."""
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
    return sorted(files)

def get_sop_files(mode: str):
    """Download source (non-QA SOP/reference txt files)."""
    if not os.path.isdir(DATA_DIR):
        return []

    files = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt"):
            continue
        if f.endswith("-QA.txt"):
            continue  # keep QA only for answering

        lf = f.lower()
        if mode == "Bayut" and not lf.startswith("bayut"):
            continue
        if mode == "dubizzle" and not lf.startswith("dubizzle"):
            continue

        # For General, include everything non-QA
        files.append(os.path.join(DATA_DIR, f))
    return sorted(files)

def pick_sop_files(mode: str, question: str):
    files = get_sop_files(mode)
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
# PARSE Q&A FILES → (Q, A) PAIRS
# ===============================
QA_PAIR_RE = re.compile(
    r"(?:^|\n)Q:\s*(.*?)\nA:\s*([\s\S]*?)(?=\nQ:\s*|\Z)",
    re.IGNORECASE
)

def parse_qa_pairs(text: str):
    pairs = []
    for m in QA_PAIR_RE.finditer(text):
        q = (m.group(1) or "").strip()
        a = (m.group(2) or "").strip()
        if q and a:
            pairs.append((q, a))
    return pairs

# ===============================
# BUILD INDEX (Q-ONLY EMBEDDINGS)
# ===============================
@st.cache_resource
def load_index(mode: str):
    qa_files = get_qa_files(mode)
    if not qa_files:
        return None

    docs = []
    for fp in qa_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(fp, "r", encoding="utf-8-sig") as f:
                raw = f.read()

        for q, a in parse_qa_pairs(raw):
            docs.append(
                Document(
                    page_content=q.strip(),  # embed ONLY the question
                    metadata={
                        "answer": a.strip(),
                        "source_file": os.path.basename(fp)
                    }
                )
            )

    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

# ===============================
# SMART ANSWERING (INDIRECT QUESTIONS OK)
# ===============================
def smart_answer(question: str, index, history, mode: str) -> str:
    q = question.strip()
    q_lower = q.lower()

    # In "Thinking" mode we softly enrich the query using last user question (if any)
    enriched = q
    if history:
        last_q = (history[-1].get("q") or "").strip()
        if last_q:
            enriched = f"{q}\nRelated context: {last_q}"

    # Fetch top matches
    results = index.similarity_search(enriched, k=5)

    if not results:
        return "I couldn’t find a clear answer to that."

    # If user is asking "who..." about corrections/updates/listings, combine key roles
    who_intent = q_lower.startswith("who") or q_lower.startswith("who ")
    correction_intent = any(k in q_lower for k in ["correct", "correction", "update", "wrong", "name", "listing", "listings"])

    if who_intent and correction_intent:
        role_answers = []
        seen = set()

        for d in results:
            qq = (d.page_content or "").strip().lower()
            ans = clean_answer((d.metadata or {}).get("answer", "")).strip()
            if not ans:
                continue

            # Prefer Qs that themselves are "Who ..."
            if qq.startswith("who"):
                key = re.sub(r"\s+", " ", ans.lower())
                if key not in seen:
                    seen.add(key)
                    role_answers.append(ans)

        if role_answers:
            # Join as a clean multi-sentence answer (no bullets forced)
            return " ".join(role_answers).strip()

    # Default: best single match
    best = results[0]
    return clean_answer((best.metadata or {}).get("answer", "")).strip() or "I couldn’t find a clear answer to that."

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
# UI – QUESTION
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    # Buttons closer together (tiny gap)
    b1, b2 = safe_columns([1, 1], gap="small")
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
    # Download SOP request → show download buttons in chat
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

    history = st.session_state.chat[tool_mode]

    if answer_mode == "Ultra-Fast":
        answer = smart_answer(q, index, [], tool_mode)
    else:
        answer = smart_answer(q, index, history, tool_mode)

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY (PER TOOL)
# ===============================
def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

style = bubble_style(tool_mode)

for idx, item in enumerate(st.session_state.chat[tool_mode]):
    q_txt = html.escape(item.get("q", ""))

    # 1) ONLY QUESTION IN THE BUBBLE
    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:6px;">
            <b>Q:</b> {q_txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2) ANSWER BELOW (NORMAL)
    a_txt = item.get("a", "")
    if a_txt:
        st.markdown(clean_answer(a_txt))

    # 3) Download buttons inside chat (only when requested)
    downloads = item.get("downloads", [])
    if downloads:
        cols = st.columns(min(3, len(downloads)))
        for i, fp in enumerate(downloads):
            col = cols[i % len(cols)]
            filename = os.path.basename(fp).replace("_", " ")

            with col:
                with open(fp, "rb") as f:
                    st.download_button(
                        label=f"Download {filename}",
                        data=f.read(),
                        file_name=os.path.basename(fp),
                        mime="text/plain",
                        key=f"dl_{tool_mode}_{idx}_{i}_{os.path.basename(fp)}"
                    )

    st.markdown("---")
