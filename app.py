cat > app.py << 'EOF'
import os
import re
import html
import time
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# UI CSS (BIGGER MODE TITLE + CLOSER TO BOX)
# ===============================
st.markdown(
    """
    <style>
      /* less gap between Ask/Clear */
      div[data-testid="stHorizontalBlock"] { gap: 0.12rem; }
      button { white-space: nowrap !important; }

      /* ✅ Mode title bigger + closer */
      .mode-title{
        font-size: 20px !important;
        font-weight: 750 !important;
        margin-top: 6px !important;
        margin-bottom: 6px !important;
        line-height: 1.2 !important;
      }

      /* ✅ Center question form and reduce top spacing */
      .question-wrap{
        max-width: 980px;
        margin: 10px auto 10px auto;   /* less space above */
      }

      .question-wrap [data-testid="stForm"]{
        border: 1px solid #E7E9EE;
        border-radius: 12px;
        padding: 16px 18px 10px 18px;
        background: #fff;
      }

      /* hide default label for text_input */
      .question-wrap div[data-testid="stTextInput"] > label { display:none; }

      hr { margin: 14px 0; }
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
# OFFLINE HF MODEL (Cloud Run Fix)
# ===============================
# Dockerfile should bake model into /models. Cloud Run needs writable cache => /tmp/models
MODEL_SRC = "/models"
MODEL_CACHE = "/tmp/models"

def _copy_baked_model_to_tmp():
    try:
        if os.path.isdir(MODEL_SRC):
            os.makedirs(MODEL_CACHE, exist_ok=True)
            # Only copy once if tmp is empty
            if not os.listdir(MODEL_CACHE):
                for item in os.listdir(MODEL_SRC):
                    s = os.path.join(MODEL_SRC, item)
                    d = os.path.join(MODEL_CACHE, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
    except Exception:
        pass

_copy_baked_model_to_tmp()

# Force all HF libs to use writable cache + offline mode
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===============================
# SESSION STATE (ROBUST + MIGRATE OLD KEYS)
# ===============================
if "chat" not in st.session_state or not isinstance(st.session_state.chat, dict):
    st.session_state.chat = {}

# migrate legacy key: "dubizzle" -> "Dubizzle"
if "dubizzle" in st.session_state.chat and "Dubizzle" not in st.session_state.chat:
    st.session_state.chat["Dubizzle"] = st.session_state.chat.pop("dubizzle")

st.session_state.chat.setdefault("General", [])
st.session_state.chat.setdefault("Bayut", [])
st.session_state.chat.setdefault("Dubizzle", [])

# ===============================
# CONSTANTS
# ===============================
THINKING_DELAY_SECONDS = 1.2

# These files are DOWNLOAD-ONLY (official SOP refs) and MUST NOT be used for Q&A answering
DOWNLOAD_ONLY_FILES = {
    "bayut-algolia locations sop.txt",
    "bayut-mybayut newsletters sop.txt",
    "bayut-pm campaigns sop.txt",
    "bayut-social media posting sop.txt",
    "both corrections and updates for listings.txt",
    "dubizzle newsletters sop.txt",
    "dubizzle pm campaigns sop.txt",
}

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
    aw = set(re.findall(r"[a-zA-Z0-9]+", (a or "").lower()))
    bw = set(re.findall(r"[a-zA-Z0-9]+", (b or "").lower()))
    return len(aw & bw)

def is_download_sop_request(question: str) -> bool:
    q = (question or "").lower().strip()
    return (("download" in q) or ("dl" in q) or ("get" in q)) and (("sop" in q) or ("sops" in q))

def bubble_style(mode: str) -> str:
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "Dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

def clean_answer(text: str) -> str:
    if not text:
        return ""

    # remove citation-like noise if present
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]*filecite[^A-Za-z0-9]*", "", text, flags=re.IGNORECASE)

    # remove Q/A prefixes
    text = re.sub(r"\bQ:\s*", "", text)
    text = re.sub(r"\bA:\s*", "", text)

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
    lines = (text or "").splitlines()
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
    return bool(re.search(r"^\s*Q(\d+)?\s*[\)\:\.]", text or "", flags=re.IGNORECASE | re.MULTILINE)) and \
           bool(re.search(r"^\s*A\s*[:\)\.]", text or "", flags=re.IGNORECASE | re.MULTILINE))

# ===============================
# INTENT RULES
# ===============================
def is_listings_intent(q: str) -> bool:
    ql = (q or "").lower()
    keys = [
        "listing", "listings", "wrong name", "wrong names", "correction", "correct",
        "update", "merge", "archive", "location", "algolia", "parent location",
        "channel handler", "chanel handler", "poc", "point of contact"
    ]
    return any(k in ql for k in keys)

def is_channel_handler_question(q: str) -> bool:
    ql = (q or "").lower()
    return ("channel handler" in ql) or ("chanel handler" in ql)

def is_poc_question(q: str) -> bool:
    ql = (q or "").lower()
    return ("poc" in ql) or ("point of contact" in ql) or ("contact person" in ql)

def is_name_question(q: str) -> bool:
    ql = (q or "").lower().strip()
    return any(p in ql for p in [
        "its name", "it's name", "what is its name", "what's its name",
        "do you know its name", "do you know it's name",
        "name?", "the name"
    ])

def is_app_owner_question(q: str) -> bool:
    ql = (q or "").lower()
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
    return sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".txt")
    ])

def mode_allows_file(mode: str, filename_lower: str) -> bool:
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
    if mode == "Dubizzle":
        return is_dubizzle or is_both or is_general
    return True

def get_qa_files(mode: str):
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if name in DOWNLOAD_ONLY_FILES:
            continue  # never use download-only files in Q&A
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
        if name in DOWNLOAD_ONLY_FILES:
            continue  # never use download-only files in Q&A
        if not mode_allows_file(mode, name):
            continue
        raw = read_text(fp)
        if not file_has_qa_pairs(raw):
            files.append(fp)
    return files

def get_download_only_sop_files(mode: str):
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if name not in DOWNLOAD_ONLY_FILES:
            continue
        if not mode_allows_file(mode, name):
            continue
        # download-only list includes SOP style files
        files.append(fp)
    return files

def pick_sop_files_for_download(mode: str, question: str):
    # ONLY download-only SOPs are shown for download requests
    files = get_download_only_sop_files(mode)
    if not files:
        return []

    stop = {"download", "dl", "get", "sop", "sops", "please", "the", "a", "an", "for", "to", "of"}
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", (question or "").lower()) if t not in stop]

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
# EMBEDDINGS (OFFLINE)
# ===============================
def _model_cache_has_files() -> bool:
    try:
        return os.path.isdir(MODEL_CACHE) and len(os.listdir(MODEL_CACHE)) > 0
    except Exception:
        return False

@st.cache_resource
def get_embeddings():
    if not _model_cache_has_files():
        # If this happens, Docker image didn't bake model correctly
        raise RuntimeError("Embedding model not found in cache. Rebuild the Docker image (model must be baked into /models).")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=MODEL_CACHE,
        model_kwargs={"local_files_only": True}
    )

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

    emb = get_embeddings()
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

    emb = get_embeddings()
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

        # never answer from download-only files (extra safety)
        if (src or "").lower() in DOWNLOAD_ONLY_FILES:
            continue

        if app_scope:
            if is_marketing_file(src):
                continue
            if not is_app_file(src):
                continue

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

def enhance_for_thinking(answer: str) -> str:
    a = clean_answer(answer)
    if not a:
        return a
    if len(a.split()) >= 60:
        return a
    return f"{a}\n\nIf you want more detail, ask: “step-by-step” or “give me the checklist”."

def smart_answer(question: str, qa_index, sop_index, answer_mode: str, history):
    # App owner answer (fixed)
    if is_app_owner_question(question):
        return "Faten Aish and Sarah Al Nawah."

    recent_qs = " ".join([h.get("q", "") for h in (history[-3:] if history else [])]).lower()
    followup_to_listings = is_listings_intent(recent_qs) and (
        is_name_question(question) or is_poc_question(question) or is_channel_handler_question(question) or
        len(re.findall(r"[a-zA-Z0-9]+", question or "")) <= 6
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

    if any(w in (question or "").lower() for w in ["assistant", "tool", "app"]):
        a = answer_from_qa(question, qa_index, answer_mode, history, force_app_filter=True)
        if a:
            return a

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
    tool_mode = st.radio("", ["General", "Bayut", "Dubizzle"], index=0)
    st.markdown("---")
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"], index=0)

chat_key = tool_mode
st.session_state.chat.setdefault(chat_key, [])

# ===============================
# MAIN TITLE
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
# CENTERED MODE TITLE + QUESTION UI
# ===============================
st.markdown('<div class="question-wrap">', unsafe_allow_html=True)

if tool_mode == "Bayut":
    st.markdown(
        '<div class="mode-title">Ask <span style="color:#0E8A6D;font-weight:800;">Bayut</span> Anything</div>',
        unsafe_allow_html=True
    )
elif tool_mode == "Dubizzle":
    st.markdown(
        '<div class="mode-title">Ask <span style="color:#D71920;font-weight:800;">Dubizzle</span> Anything</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="mode-title">General Assistant</div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here...")

    left, mid, right = st.columns([4, 2.3, 4])
    with mid:
        b1, b2 = st.columns([1, 1], gap="small")
        ask = b1.form_submit_button("Ask")
        clear = b2.form_submit_button("Clear chat")

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# CLEAR CHAT
# ===============================
if clear:
    st.session_state.chat[chat_key] = []
    st.rerun()

# ===============================
# HANDLE QUESTION
# ===============================
def _safe_load_indexes(mode: str):
    try:
        qa_index = load_qa_index(mode)
        sop_index = load_sop_index(mode)
        return qa_index, sop_index, None
    except Exception as e:
        return None, None, str(e)

if ask and (q or "").strip():
    history = st.session_state.chat.get(chat_key, [])

    # Download-only SOP request => show download buttons in chat
    if is_download_sop_request(q):
        files = pick_sop_files_for_download(chat_key, q)
        if not files:
            st.session_state.chat[chat_key].append({"q": q, "a": "No SOP files found to download.", "downloads": []})
        else:
            st.session_state.chat[chat_key].append({"q": q, "a": "Here are the SOP files you can download:", "downloads": files})
        st.rerun()

    if answer_mode == "Thinking":
        with st.spinner("Thinking..."):
            qa_index, sop_index, err = _safe_load_indexes(chat_key)
            if err:
                st.session_state.chat[chat_key].append({"q": q, "a": f"Error: {err}"})
                st.rerun()

            answer = smart_answer(q, qa_index, sop_index, answer_mode, history)
            time.sleep(THINKING_DELAY_SECONDS)
            answer = enhance_for_thinking(answer)

            st.session_state.chat[chat_key].append({"q": q, "a": answer})
            st.rerun()
    else:
        qa_index, sop_index, err = _safe_load_indexes(chat_key)
        if err:
            st.session_state.chat[chat_key].append({"q": q, "a": f"Error: {err}"})
            st.rerun()

        answer = smart_answer(q, qa_index, sop_index, answer_mode, history)
        st.session_state.chat[chat_key].append({"q": q, "a": answer})
        st.rerun()

# ===============================
# CHAT HISTORY (NEWEST ON TOP)
# ===============================
style = bubble_style(chat_key)
items = list(enumerate(st.session_state.chat.get(chat_key, [])))[::-1]

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
                        key=f"dl_{chat_key}_{orig_idx}_{i}_{os.path.basename(fp)}"
                    )

    st.markdown("---")
EOF
