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

# ✅ UI: bold label + less space between buttons
st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { gap: 0.14rem; }
      button { white-space: nowrap !important; }
      .ask-label { font-weight: 800; margin-bottom: 6px; }
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
def read_text(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8-sig") as f:
            return f.read()

def keyword_overlap(a: str, b: str) -> int:
    aw = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    bw = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    return len(aw & bw)

def is_download_sop_request(question: str) -> bool:
    q = question.lower().strip()
    return (("download" in q) or ("dl" in q) or ("get" in q)) and (("sop" in q) or ("sops" in q))

def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

def clean_answer(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"", "", text, flags=re.DOTALL)
    text = re.sub(r"", "", text, flags=re.DOTALL)

    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]*filecite[^A-Za-z0-9]*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\bQ:\s*", "", text)
    text = re.sub(r"\bA:\s*", "", text)
    text = re.sub(r"\bQ\d+\)\s*", "", text, flags=re.IGNORECASE)

    text = text.replace("*", " ")
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[•\-\u2022]\s*", "", line)
        if line:
            lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# ===============================
# FLEXIBLE Q/A PARSER (Q:/A: + Q1)/A:)
# ===============================
Q_LINE_RE = re.compile(r"^\s*Q(\d+)?\s*[\)\:\.\-]?\s*(.+?)\s*$", re.IGNORECASE)
A_LINE_RE = re.compile(r"^\s*A\s*[\)\:\.\-]?\s*(.*)\s*$", re.IGNORECASE)

def parse_qa_pairs_flexible(text: str):
    lines = text.splitlines()
    pairs = []

    cur_q = None
    cur_a_lines = []
    in_answer = False

    def flush():
        nonlocal cur_q, cur_a_lines, in_answer
        if cur_q and cur_a_lines:
            a = "\n".join(cur_a_lines).strip()
            if a:
                pairs.append((cur_q.strip(), a))
        cur_q = None
        cur_a_lines = []
        in_answer = False

    for line in lines:
        qmatch = Q_LINE_RE.match(line)
        if qmatch:
            flush()
            cur_q = qmatch.group(2).strip()
            continue

        amatch = A_LINE_RE.match(line)
        if amatch and cur_q:
            in_answer = True
            first = amatch.group(1).strip()
            if first:
                cur_a_lines.append(first)
            continue

        if in_answer and cur_q:
            cur_a_lines.append(line.rstrip())

    flush()
    return pairs

def file_has_qa_pairs(text: str) -> bool:
    return bool(re.search(r"^\s*Q(\d+)?\s*[\)\:\.]", text, flags=re.IGNORECASE | re.MULTILINE)) and \
           bool(re.search(r"^\s*A\s*[:\)\.]", text, flags=re.IGNORECASE | re.MULTILINE))

# ===============================
# INTENT RULES
# ===============================
def is_listings_intent(q: str) -> bool:
    ql = q.lower()
    keys = [
        "listing", "listings", "wrong name", "wrong names", "correction", "correct",
        "update", "merge", "archive", "location", "algolia", "parent location",
        "channel handler", "chanel handler", "poc", "point of contact"
    ]
    return any(k in ql for k in keys)

def is_channel_handler_question(q: str) -> bool:
    ql = q.lower()
    return ("channel handler" in ql) or ("chanel handler" in ql)

def is_poc_question(q: str) -> bool:
    ql = q.lower()
    return ("poc" in ql) or ("point of contact" in ql) or ("contact person" in ql)

def is_name_question(q: str) -> bool:
    ql = q.lower().strip()
    return any(p in ql for p in [
        "its name", "it's name", "what is its name", "what's its name",
        "do you know its name", "do you know it's name",
        "name?", "the name"
    ])

# ✅ NEW: APP/TOOL OWNERSHIP INTENT
def is_app_owner_question(q: str) -> bool:
    ql = q.lower()
    keys = [
        "who is responsible for the app",
        "who is resposible for the app",
        "responsible for the app",
        "who is responsible for this tool",
        "who is resposible about this tool",
        "who owns the app",
        "app owner",
        "tool owner",
        "who maintains the app",
        "who runs the app",
        "who is responsible about this tool",
    ]
    return any(k in ql for k in keys)

# ===============================
# FILE SELECTION
# ===============================
MARKETING_BLOCK = ["pm", "campaign", "newsletter", "social", "paid", "performance", "design team"]
LISTINGS_ALLOW = ["correction", "update", "listing", "listings", "location", "algolia", "projects", "project", "corrections"]
APP_ALLOW = ["assistant", "internal ai", "ai content assistant", "app", "tool", "onboarding", "faq", "help"]

def list_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])

def mode_allows_file(mode: str, filename_lower: str) -> bool:
    # ✅ Always allow “app/tool/assistant/faq” files in any mode
    if any(k in filename_lower for k in APP_ALLOW):
        return True

    is_bayut = filename_lower.startswith("bayut")
    is_dubizzle = filename_lower.startswith("dubizzle")
    is_both = filename_lower.startswith("both")
    is_general = filename_lower.startswith("general")

    if mode == "General":
        return True
    if mode == "Bayut":
        return is_bayut or is_both or is_general
    if mode == "dubizzle":
        return is_dubizzle or is_both or is_general
    return True

def get_qa_files(mode: str):
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if not mode_allows_file(mode, name):
            continue
        raw = read_text(fp)
        if file_has_qa_pairs(raw):
            files.append(fp)
    return files

def get_sop_files(mode: str):
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if not mode_allows_file(mode, name):
            continue
        raw = read_text(fp)
        if not file_has_qa_pairs(raw):
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
    best = scored[0][0]
    if best == 0:
        return files
    return [fp for s, fp in scored if s == best]

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
        for q, a in parse_qa_pairs_flexible(raw):
            docs.append(Document(
                page_content=q.strip(),
                metadata={"answer": a.strip(), "source_file": os.path.basename(fp).lower()}
            ))

    if not docs:
        return None

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

@st.cache_resource
def load_sop_index(mode: str):
    sop_files = get_sop_files(mode)
    if not sop_files:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=140)
    docs = []
    for fp in sop_files:
        raw = read_text(fp)
        for c in splitter.split_text(raw):
            docs.append(Document(page_content=c, metadata={"source_file": os.path.basename(fp).lower()}))

    if not docs:
        return None

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

# ===============================
# ANSWERING
# ===============================
def is_marketing_file(name_lower: str) -> bool:
    nl = (name_lower or "").lower()
    return any(k in nl for k in MARKETING_BLOCK)

def is_listings_file(name_lower: str) -> bool:
    nl = (name_lower or "").lower()
    return any(k in nl for k in LISTINGS_ALLOW) or nl.startswith("both")

def is_app_file(name_lower: str) -> bool:
    nl = (name_lower or "").lower()
    return any(k in nl for k in APP_ALLOW)

def answer_from_qa(question: str, qa_index, answer_mode: str, history,
                   force_listings_filter: bool = False,
                   force_app_filter: bool = False):
    if qa_index is None:
        return None

    enriched = question
    if answer_mode == "Thinking" and history:
        last_q = (history[-1].get("q") or "").strip()
        if last_q:
            enriched = f"{question}\nRelated: {last_q}"

    results = qa_index.similarity_search(enriched, k=12)
    if not results:
        return None

    listings_scope = is_listings_intent(question) or force_listings_filter
    app_scope = is_app_owner_question(question) or force_app_filter

    filtered = []
    for d in results:
        src = (d.metadata or {}).get("source_file", "")

        # ✅ APP scope: only allow app/assistant/faq files (block marketing completely)
        if app_scope:
            if is_marketing_file(src):
                continue
            if not is_app_file(src):
                continue

        # ✅ Listings scope: only allow listings/corrections files (block marketing completely)
        if listings_scope:
            if is_marketing_file(src):
                continue
            if not is_listings_file(src):
                continue

        filtered.append(d)

    if not filtered:
        return None

    best_doc = None
    best_score = -1
    for d in filtered:
        ov = keyword_overlap(question, d.page_content)
        if ov > best_score:
            best_score = ov
            best_doc = d

    if best_doc is None:
        return None

    ans = clean_answer((best_doc.metadata or {}).get("answer", ""))
    return ans if ans else None

def answer_from_sop_fallback(question: str, sop_index):
    if sop_index is None:
        return None
    res = sop_index.similarity_search(question, k=4)
    if res:
        return clean_answer(res[0].page_content)[:450]
    return None

def smart_answer(question: str, qa_index, sop_index, answer_mode: str, history):
    # ✅ Hard override: app ownership ALWAYS returns correct answer
    if is_app_owner_question(question):
        return "Faten Aish and Sarah Al Nawah."

    # follow-up lock: listings thread
    recent_qs = " ".join([h.get("q", "") for h in (history[-3:] if history else [])]).lower()
    followup_to_listings = is_listings_intent(recent_qs) and (
        is_name_question(question) or is_poc_question(question) or is_channel_handler_question(question) or
        len(re.findall(r"[a-zA-Z0-9]+", question)) <= 6
    )

    listings_scope = is_listings_intent(question) or followup_to_listings

    if listings_scope:
        a = answer_from_qa(question, qa_index, answer_mode, history, force_listings_filter=True)
        if a:
            return a
        b = answer_from_sop_fallback(question, sop_index)
        if b:
            return b
        return "I couldn’t find a clear answer to that."

    # ✅ App/tool general questions (non-owner): search only app/faq files first
    if "assistant" in question.lower() or "tool" in question.lower() or "app" in question.lower():
        a = answer_from_qa(question, qa_index, answer_mode, history, force_app_filter=True)
        if a:
            return a

    # normal QA
    a = answer_from_qa(question, qa_index, answer_mode, history)
    if a:
        return a

    b = answer_from_sop_fallback(question, sop_index)
    if b:
        return b

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
    st.markdown('<div class="ask-label">Ask a question</div>', unsafe_allow_html=True)
    q = st.text_input("", placeholder="Type your question here...")

    left, mid, right = st.columns([4, 2.3, 4])
    with mid:
        b1, b2 = st.columns([1, 1], gap="small")
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
# CHAT HISTORY (NEWEST ON TOP)
# ===============================
style = bubble_style(tool_mode)
items = list(enumerate(st.session_state.chat[tool_mode]))[::-1]

for orig_idx, item in items:
    q_txt = html.escape(item.get("q", ""))

    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:6px;">
            <b>Q:</b> {q_txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    a_txt = clean_answer(item.get("a", ""))
    if a_txt:
        st.markdown(a_txt)

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
                        key=f"dl_{tool_mode}_{orig_idx}_{i}_{os.path.basename(fp)}"
                    )

    st.markdown("---")
