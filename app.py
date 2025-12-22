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

# ✅ Fix button wrapping + keep small gap (natural look)
st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
      button { white-space: nowrap !important; }  /* prevents A\ns\nk */
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

def is_listings_intent(q: str) -> bool:
    ql = q.lower()
    keys = [
        "listing", "listings", "wrong name", "wrong names", "correction", "correct",
        "update", "merge", "archive", "location", "algolia", "parent location"
    ]
    return any(k in ql for k in keys)

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
    """
    ✅ Natural output cleanup:
    - removes citations + turnXfileY artifacts
    - removes control chars
    - removes bullets and asterisks
    - returns clean paragraph(s)
    """
    if not text:
        return ""

    # Remove private-use citation blocks
    text = re.sub(r"", "", text, flags=re.DOTALL)
    text = re.sub(r"", "", text, flags=re.DOTALL)

    # Remove turnXfileY artifacts
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"turn\d+file\d+", "", text, flags=re.IGNORECASE)

    # Remove filecite word even if surrounded by weird symbols
    text = re.sub(r"[^A-Za-z0-9]*filecite[^A-Za-z0-9]*", "", text, flags=re.IGNORECASE)

    # Remove repeated Q:/A: tokens inside answers
    text = re.sub(r"\bQ:\s*", "", text)
    text = re.sub(r"\bA:\s*", "", text)

    # Remove asterisks bullets like "* Verify ..."
    text = text.replace("*", " ")

    # Strip control characters
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # Flatten bullets/dashes into one paragraph
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
# Q/A PARSER (only when file contains Q:/A:)
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
# FILE SELECTION
# ===============================
MARKETING_BLOCK = ["pm", "campaign", "newsletter", "social", "paid", "performance", "design team"]
LISTINGS_ALLOW = ["correction", "update", "listing", "listings", "location", "algolia"]

def list_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])

def mode_allows_file(mode: str, filename_lower: str) -> bool:
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
    """Files that actually contain Q:/A: pairs (regardless of filename)."""
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if not mode_allows_file(mode, name):
            continue
        raw = read_text(fp)
        if "Q:" in raw and "\nA:" in raw:
            files.append(fp)
    return files

def get_sop_files(mode: str):
    """Plain SOP files without Q/A (includes Both Corrections...txt)."""
    files = []
    for fp in list_txt_files():
        name = os.path.basename(fp).lower()
        if not mode_allows_file(mode, name):
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
        for q, a in parse_qa_pairs(raw):
            docs.append(Document(
                page_content=q.strip(),  # embed ONLY question
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
        chunks = splitter.split_text(raw)
        for c in chunks:
            docs.append(Document(
                page_content=c,
                metadata={"source_file": os.path.basename(fp).lower()}
            ))

    if not docs:
        return None

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

# ===============================
# ANSWERING (FIXED ROUTER + NATURAL LISTINGS OUTPUT)
# ===============================
def is_marketing_file(name_lower: str) -> bool:
    nl = (name_lower or "").lower()
    return any(k in nl for k in MARKETING_BLOCK)

def is_listings_file(name_lower: str) -> bool:
    nl = (name_lower or "").lower()
    return any(k in nl for k in LISTINGS_ALLOW) or nl.startswith("both")

def extract_sentences(text: str):
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def answer_from_sop_listings_first(question: str, sop_index):
    """
    ✅ Listings intent:
    - only look at BOTH/listings/corrections files
    - return a clean, natural 1–2 sentence answer (not raw bullets)
    """
    if sop_index is None:
        return None

    results = sop_index.similarity_search(question, k=12)
    if not results:
        return None

    filtered = []
    for d in results:
        src = (d.metadata or {}).get("source_file", "")
        if is_marketing_file(src):
            continue
        if is_listings_file(src):
            filtered.append(d)

    if not filtered:
        return None

    # Build a small evidence text from top chunks
    evidence = " ".join((d.page_content or "") for d in filtered[:3])
    ev = clean_answer(evidence).lower()

    ops = "operations" in ev
    handler = ("channel handler" in ev) or ("coordination" in ev)
    submit = ("submit" in ev) or ("request" in ev) or ("content team" in ev) or ("content teams" in ev)

    # Natural answer templates
    if submit and ops and handler:
        return ("The Content team submits the correction request, Operations updates the listing data once it’s approved, "
                "and the Channel Handler verifies the change live on the platform.")
    if ops and handler:
        return ("Operations updates the listing data after approval, and the Channel Handler verifies the change live on the platform.")
    if ops:
        return "Operations updates the listing data after the correction is approved."
    if handler:
        return "The Channel Handler verifies the change live on the platform after it’s updated."

    # Fallback: pick best 2 sentences
    cand = []
    for d in filtered[:4]:
        for s in extract_sentences(d.page_content or ""):
            ov = keyword_overlap(question, s)
            role_push = 1 if any(w in s.lower() for w in ["operations", "coordination", "channel handler", "verify", "submit", "request", "backend"]) else 0
            cand.append((ov + role_push, s))

    cand.sort(key=lambda x: x[0], reverse=True)
    best = [c[1] for c in cand[:2] if c[0] > 0]
    if best:
        return clean_answer(" ".join(best))

    return clean_answer(filtered[0].page_content)[:350]

def answer_from_qa(question: str, qa_index, answer_mode: str, history):
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

    listings_intent = is_listings_intent(question)

    filtered = []
    for d in results:
        src = (d.metadata or {}).get("source_file", "")
        if listings_intent:
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

    if best_doc is None or best_score <= 0:
        return None

    ans = clean_answer((best_doc.metadata or {}).get("answer", ""))
    return ans if ans else None

def smart_answer(question: str, qa_index, sop_index, answer_mode: str, history):
    # Listings intent → SOP listings first → QA listings only
    if is_listings_intent(question):
        a = answer_from_sop_listings_first(question, sop_index)
        if a:
            return a
        a = answer_from_qa(question, qa_index, answer_mode, history)
        if a:
            return a
        return "I couldn’t find a clear answer to that."

    # Non-listings questions → QA first, then SOP fallback
    a = answer_from_qa(question, qa_index, answer_mode, history)
    if a:
        return a

    if sop_index is not None:
        res = sop_index.similarity_search(question, k=4)
        if res:
            return clean_answer(res[0].page_content)[:450]

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

    # ✅ FIX: keep buttons left + close + wide enough to avoid vertical letters
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
# CHAT HISTORY
# ===============================
style = bubble_style(tool_mode)

for idx, item in enumerate(st.session_state.chat[tool_mode]):
    q_txt = html.escape(item.get("q", ""))

    # Only question in bubble
    st.markdown(
        f"""
        <div style="{style} padding:12px;border-radius:10px;margin-bottom:6px;">
            <b>Q:</b> {q_txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Answer below (cleaned + natural)
    a_txt = clean_answer(item.get("a", ""))
    if a_txt:
        st.markdown(a_txt)

    # Download buttons inside chat
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
