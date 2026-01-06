import os
import re
import html
import time
import hashlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# =====================================================
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # put your .txt here in the repo

# =====================================================
# STYLES (KEEP YOUR DESIGN)
# =====================================================
st.markdown(
    """
    <style>
      section.main > div.block-container{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
      }

      div[data-testid="stCaptionContainer"]{ display:none !important; }
      div[data-testid="stHeader"]{ background: transparent !important; }
      header{ background: transparent !important; }

      #brand-header{ text-align:center !important; margin: 0 0 10px 0 !important; }
      #brand-title{
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        line-height: 1.15 !important;
      }
      #brand-subtitle{
        margin-top: 6px !important;
        font-size: 1.05rem !important;
        opacity: 0.75 !important;
      }

      div.stButton > button{
        border-radius: 12px !important;
        font-weight: 600 !important;
      }

      .ask-label{
        font-weight: 800 !important;
        font-size: 1rem !important;
        margin: 12px 0 6px 0 !important;
      }

      .q-bubble{
        padding: 12px 16px !important;
        border-radius: 16px !important;
        max-width: 85% !important;
        width: fit-content !important;
        font-weight: 700 !important;
        margin: 12px 0 8px 0 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        background: #ffffff !important;
      }

      .a-bubble{
        padding: 12px 16px !important;
        border-radius: 16px !important;
        max-width: 92% !important;
        width: fit-content !important;
        margin: 6px 0 18px 6px !important;
        line-height: 1.7 !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        white-space: normal !important;
      }
      .a-general{ background:#f2f2f2 !important; }
      .a-bayut{ background:#e6f4ef !important; border-color: rgba(14,138,109,0.22) !important; }
      .a-dubizzle{ background:#fdeaea !important; border-color: rgba(215,25,32,0.22) !important; }

      .mode-title{
        text-align:center !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 18px 0 10px 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("answer_mode", "Ultra-Fast")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# =====================================================
# HELPERS
# =====================================================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def br(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")

def is_sop_file(name: str) -> bool:
    # Anything with SOP in filename is excluded
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    """
    Your rule:
    - General mode answers only from General QA files
    - Bayut mode answers only from Bayut/MyBayut files
    - dubizzle mode answers only from dubizzle files
    """
    n = name.lower()

    # MyBayut is Bayut content
    if "mybayut" in n:
        return "Bayut"

    has_bayut = "bayut" in n
    has_dub = "dubizzle" in n

    # cross-brand or general
    if has_bayut and has_dub:
        return "General"
    if any(x in n for x in ["general", "both", "common", "shared", "cross"]):
        return "General"

    if has_bayut:
        return "Bayut"
    if has_dub:
        return "Dubizzle"

    # default safe
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text: str):
    pattern = re.compile(r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)", re.S | re.I)
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def merge_unique(items: list[str], limit: int = 6) -> str:
    out = []
    for x in items:
        x = (x or "").strip()
        if x and x not in out:
            out.append(x)
    return "\n\n".join(out[:limit])

def is_entity_question(q: str) -> bool:
    qn = normalize(q)
    return qn.startswith(("who is", "what is", "من هو", "من هي", "ما هو", "ما هي"))

def extract_entity(q: str) -> str:
    qn = normalize(q)
    qn = re.sub(r"^(who is|what is)\s+", "", qn).strip()
    qn = re.sub(r"^(من هو|من هي|ما هو|ما هي)\s+", "", qn).strip()
    qn = re.sub(r"[?!.]+$", "", qn).strip()
    return qn

def token_in_text(text: str, token: str) -> bool:
    t = normalize(text)
    tok = normalize(token)
    if not tok:
        return False
    return re.search(rf"(^|[^a-z0-9_]){re.escape(tok)}([^a-z0-9_]|$)", t) is not None

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# AUTO REFRESH WHEN DATA FILES CHANGE (NO REDEPLOY NEEDED)
# =====================================================
def file_sig(path: str) -> str:
    # fast signature: mtime + size (enough to trigger rebuild)
    st_ = os.stat(path)
    return f"{int(st_.st_mtime)}:{st_.st_size}"

def data_fingerprint():
    if not os.path.isdir(DATA_DIR):
        return tuple()
    items = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        try:
            items.append((f, file_sig(fp)))
        except Exception:
            pass
    return tuple(sorted(items))

# =====================================================
# BUILD STORES (STRICT MODE + EMBED Q+A)
# =====================================================
@st.cache_resource
def build_assets(_fingerprint):
    if not os.path.isdir(DATA_DIR):
        return {"vs": {"General": None, "Bayut": None, "Dubizzle": None}, "docs": {"General": [], "Bayut": [], "Dubizzle": []}}

    emb = get_embeddings()
    docs = {"General": [], "Bayut": [], "Dubizzle": []}

    seen = set()  # dedupe duplicates
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        mode = bucket_from_filename(f)
        fp = os.path.join(DATA_DIR, f)
        text = read_text(fp)

        for q, a in parse_qa_pairs(text):
            key = (mode, normalize(q), normalize(a))
            if key in seen:
                continue
            seen.add(key)

            # IMPORTANT: embed BOTH Q + A so "who is faten" can match answers that contain "Faten"
            content = f"Q: {q}\nA: {a}"
            docs[mode].append(
                Document(
                    page_content=content,
                    metadata={"question": q, "answer": a, "file": f, "mode": mode},
                )
            )

    vs = {
        m: (FAISS.from_documents(docs[m], emb) if docs[m] else None)
        for m in ["General", "Bayut", "Dubizzle"]
    }
    return {"vs": vs, "docs": docs}

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div id="brand-header">
      <div id="brand-title">
        <span style="color:#0E8A6D;">Bayut</span> &
        <span style="color:#D71920;">dubizzle</span>
        AI Content Assistant
      </div>
      <div id="brand-subtitle">Your Internal AI Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# TOOL MODE BUTTONS
# =====================================================
tool_cols = st.columns([2, 3, 3, 3, 2])
with tool_cols[1]:
    if st.button("General", use_container_width=True):
        st.session_state.tool_mode = "General"
with tool_cols[2]:
    if st.button("Bayut", use_container_width=True):
        st.session_state.tool_mode = "Bayut"
with tool_cols[3]:
    if st.button("dubizzle", use_container_width=True):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(f"<div class='mode-title'>{st.session_state.tool_mode} Assistant</div>", unsafe_allow_html=True)

# =====================================================
# ANSWER MODE BUTTONS
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True):
        st.session_state.answer_mode = "Thinking"

# =====================================================
# REFRESH KNOWLEDGE (NO REDEPLOY, NO REBOOT)
# =====================================================
refresh_cols = st.columns([6, 2, 6])
with refresh_cols[1]:
    refresh = st.button("Refresh Knowledge", use_container_width=True)

fp = data_fingerprint()
if refresh:
    st.cache_resource.clear()

ASSETS = build_assets(fp)

def pick_store_and_docs():
    mode = st.session_state.tool_mode
    return ASSETS["vs"].get(mode), ASSETS["docs"].get(mode, [])

# =====================================================
# INPUT (FORM = nicer, no extra rerun while typing)
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", label_visibility="collapsed", key="q_input")
    btn_cols = st.columns([1, 1])
    ask = btn_cols[0].form_submit_button("Ask", use_container_width=True)
    clear = btn_cols[1].form_submit_button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []

# =====================================================
# ANSWER (CHATGPT STYLE)
# =====================================================
if ask and q:
    vs, docs_mode = pick_store_and_docs()
    thinking = st.session_state.answer_mode == "Thinking"

    if not docs_mode:
        final = "No internal Q&A data found for this mode."
    else:
        # Entity: "who is X" => search + scan answers for X
        if is_entity_question(q):
            entity = extract_entity(q)
            ent = normalize(entity)

            variants = {ent}
            parts = [p for p in ent.split(" ") if p]
            if parts:
                variants.add(parts[0])      # first token
                variants.add(parts[-1])     # last token

            candidates = []
            if vs:
                candidates += vs.similarity_search(entity, k=30)
                candidates += vs.similarity_search(q, k=20)

            hits = []
            pool = candidates if candidates else docs_mode
            for d in pool:
                qq = d.metadata.get("question", "")
                aa = d.metadata.get("answer", "")
                if any(token_in_text(qq, v) or token_in_text(aa, v) for v in variants):
                    hits.append(aa)
                if len(hits) >= 6:
                    break

            final = merge_unique(hits) if hits else f'I couldn’t find information about “{entity}” in this mode’s knowledge files.'

        # Normal question => retrieve best answer
        else:
            if not vs:
                final = "No searchable index found for this mode."
            else:
                if thinking:
                    with st.spinner("Thinking…"):
                        time.sleep(0.15)

                results = vs.similarity_search(q, k=8 if thinking else 4)
                answers = [r.metadata.get("answer") for r in results if r.metadata.get("answer")]
                final = answers[0] if answers else "No relevant answer found."

    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})

# =====================================================
# CHAT HISTORY (ANSWER bubble changes by mode)
# =====================================================
answer_class = {
    "General": "a-general",
    "Bayut": "a-bayut",
    "Dubizzle": "a-dubizzle",
}[st.session_state.tool_mode]

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{br(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='a-bubble {answer_class}'>{br(item['a'])}</div>", unsafe_allow_html=True)
    st.markdown("---")
