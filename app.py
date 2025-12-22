import os
import re
import html
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & dubizzle Internal Assistant",
    layout="wide"
)

# Reduce spacing between columns (buttons closer)
st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { gap: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True
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
# HELPERS
# ===============================
def safe_columns(spec, gap="small"):
    try:
        return st.columns(spec, gap=gap)
    except TypeError:
        return st.columns(spec)

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

def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def clean_answer(text: str) -> str:
    """Remove filecite junk + hidden control chars; return plain readable text (no bullets)."""
    if not text:
        return ""

    # Remove private-use citation blocks like: 
    text = re.sub(r"", "", text, flags=re.DOTALL)
    text = re.sub(r"", "", text, flags=re.DOTALL)

    # Remove ANY occurrence of filecite even if surrounded by weird boxes
    text = re.sub(r"[^A-Za-z0-9]*filecite[^A-Za-z0-9]*", "", text, flags=re.IGNORECASE)

    # Strip control chars
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # Flatten bullets into one clean paragraph
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[•\-\u2022]\s*", "", line)
        if line:
            lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def is_download_sop_request(question: str) -> bool:
    q = question.lower().strip()
    return (("download" in q) or ("dl" in q) or ("get" in q)) and (("sop" in q) or ("sops" in q))

def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

def topic_bonus(question_lower: str, source_file_lower: str) -> int:
    listing_intent = any(k in question_lower for k in [
        "listing", "listings", "wrong name", "wrong names", "correction", "correct", "update",
        "merge", "archive", "location", "algolia"
    ])

    if listing_intent:
        if any(k in source_file_lower for k in ["correction", "update", "listing", "listings", "location", "algolia"]):
            return 4
        if any(k in source_file_lower for k in ["pm", "campaign", "newsletter", "social", "paid"]):
            return -4
    return 0

def keyword_overlap(a: str, b: str) -> int:
    aw = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    bw = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    return len(aw & bw)

# ===============================
# FILE SELECTION
# ===============================
def list_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])

def get_qa_files(mode: str):
    """
    QA sources are ONLY files that contain Q:/A: pairs.
    (They might be named -QA.txt OR not — we detect by content.)
    """
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()

        # mode filtering
        is_bayut = name.startswith("bayut")
        is_dubizzle = name.startswith("dubizzle")
        is_both = name.startswith("both")
        is_general = name.startswith("general")

        if mode == "Bayut" and not (is_bayut or is_both or is_general):
            continue
        if mode == "dubizzle" and not (is_dubizzle or is_both or is_general):
            continue
        # General = everything

        raw = read_text(fp)
        if "Q:" in raw and "\nA:" in raw:
            files.append(fp)

    return files

def get_sop_files(mode: str):
    """
    SOP sources are plain text files WITHOUT Q:/A: pairs.
    We MUST include Both* for Bayut and dubizzle.
    """
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()

        is_bayut = name.startswith("bayut")
        is_dubizzle = name.startswith("dubizzle")
        is_both = name.startswith("both")
        is_general = name.startswith("general")

        if mode == "Bayut" and not (is_bayut or is_both or is_general):
            continue
        if mode == "dubizzle" and not (is_dubizzle or is_both or is_general):
            continue

        raw = read_text(fp)
        if not ("Q:" in raw and "\nA:" in raw):
            files.append(fp)

    return files

def pick_sop_files_for_download(mode: str, question: str):
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
# BUILD INDICES
# ===============================
@st.cache_resource
def load_qa_index(mode: str):
    qa_files = get_qa_files(mode)
    if not qa_files:
        return None

    docs = []
    for fp in qa_files:
        raw = read_text(fp)
        for q, a in parse_qa_pairs(raw):
            docs.append(Document(
                page_content=q.strip(),  # embed question only
                metadata={
                    "answer": a.strip(),
                    "source_file": os.path.basename(fp).lower()
                }
            ))

    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource
def load_sop_index(mode: str):
    sop_files = get_sop_files(mode)
    if not sop_files:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    docs = []

    for fp in sop_files:
        raw = read_text(fp)
        chunks = splitter.split_text(raw)
        for c in chunks:
            docs.append(Document(
                page_content=c,
                metadata={"source_file": os.path.basename(fp).lower()}
            ))

    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ===============================
# ANSWERING
# ===============================
def answer_from_qa(question: str, qa_index, answer_mode: str, history):
    enriched = question
    if answer_mode == "Thinking" and history:
        last_q = (history[-1].get("q") or "").strip()
        if last_q:
            enriched = f"{question}\nRelated: {last_q}"

    # get more candidates then rerank by filename bonus
    results = qa_index.similarity_search(enriched, k=10)
    if not results:
        return None

    q_lower = question.lower()
    best_doc = None
    best_score = -10_000

    for d in results:
        src = (d.metadata or {}).get("source_file", "")
        bonus = topic_bonus(q_lower, src)
        overlap = keyword_overlap(question, d.page_content)  # overlap with matched stored Q
        score = bonus * 10 + overlap  # strong bias to correct file/topic + some lexical sanity

        if score > best_score:
            best_score = score
            best_doc = d

    if not best_doc:
        return None

    ans = clean_answer((best_doc.metadata or {}).get("answer", ""))
    if not ans:
        return None

    # If overlap is extremely low, treat as weak match (fallback to SOP)
    if keyword_overlap(question, best_doc.page_content) == 0:
        return None

    return ans

def extract_sentences(text: str):
    # Simple sentence split that works OK for SOPs
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def answer_from_sop(question: str, sop_index):
    results = sop_index.similarity_search(question, k=8)
    if not results:
        return None

    q_lower = question.lower()

    # prefer chunks from "both corrections/listings" etc.
    scored = []
    for d in results:
        src = (d.metadata or {}).get("source_file", "")
        bonus = topic_bonus(q_lower, src)
        scored.append((bonus, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top 2 chunks after bonus
    top_docs = [d for _, d in scored[:2]]

    # Extract best sentences by overlap
    cand = []
    for d in top_docs:
        for s in extract_sentences(d.page_content):
            ov = keyword_overlap(question, s)
            # extra push for role words
            role_push = 1 if any(w in s.lower() for w in ["operations", "coordination", "channel handler", "verify", "submit", "request"]) else 0
            cand.append((ov + role_push, s))

    cand.sort(key=lambda x: x[0], reverse=True)
    best = [c[1] for c in cand[:3] if c[0] > 0]

    if not best:
        # fallback: just return cleaned top chunk start
        return clean_answer(top_docs[0].page_content)[:400]

    return clean_answer(" ".join(best))

def smart_answer(question: str, qa_index, sop_index, answer_mode: str, history):
    # 1) try QA first (most accurate)
    if qa_index is not None:
        a = answer_from_qa(question, qa_index, answer_mode, history)
        if a:
            return a

    # 2) fallback to SOP retrieval (this is what fixes "Both Corrections..." file)
    if sop_index is not None:
        a = answer_from_sop(question, sop_index)
        if a:
            return a

    return "I couldn’t find a clear answer to that."

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
# TOOL HEADING
# ===============================
if tool_mode == "Bayut":
    st.subheader("Ask Bayut Anything")
elif tool_mode == "dubizzle":
    st.subheader("Ask dubizzle Anything")
else:
    st.subheader("General Assistant")

# ===============================
# QUESTION UI
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    # Buttons LEFT beside each other (tiny gap) + spacer
    b1, b2, _sp = safe_columns([1, 1, 10], gap="small")
    ask = b1.form_submit_button("Ask")
    clear = b2.form_submit_button("Clear chat")

# ===============================
# CLEAR CHAT
# ===============================
if clear:
    st.session_state.chat[tool_mode] = []
    st.rerun()

# ===============================
# ANSWER
# ===============================
if ask and q.strip():
    # download SOP in chat
    if is_download_sop_request(q):
        files = pick_sop_files_for_download(tool_mode, q)
        if not files:
            st.session_state.chat[tool_mode].append({"q": q, "a": "No SOP files found to download.", "downloads": []})
        else:
            st.session_state.chat[tool_mode].append({"q": q, "a": "Here are the SOP files you can download:", "downloads": files})
        st.rerun()

    qa_index = load_qa_index(tool_mode)
    sop_index = load_sop_index(tool_mode)

    history = st.session_state.chat[tool_mode]
    answer = smart_answer(q, qa_index, sop_index, answer_mode, history)

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT HISTORY
# ===============================
style = bubble_style(tool_mode)

for idx, item in enumerate(st.session_state.chat[tool_mode]):
    q_txt = html.escape(item.get("q", ""))

    # Only the question in the bubble
    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:6px;">
            <b>Q:</b> {q_txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Answer below (cleaned, no bullets)
    a_txt = clean_answer(item.get("a", ""))
    if a_txt:
        st.markdown(a_txt)

    # Download buttons inside chat (only when requested)
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
