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
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================
# HARD-FORCE STYLES (VERY SPECIFIC + !important)
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

      section.main h1{
        text-align:center !important;
        margin-bottom: 6px !important;
      }

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
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def is_sop_file(name: str) -> bool:
    # keep your rule (anything with "sop" is excluded)
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "bayut" in n:
        return "Bayut"
    if "dubizzle" in n:
        return "Dubizzle"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def parse_qa_pairs(text: str):
    pattern = re.compile(
        r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)",
        re.S | re.I
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(text)]

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = [primary] + extras
    cleaned = []
    for x in out:
        x = (x or "").strip()
        if x and x not in cleaned:
            cleaned.append(x)
    return "\n\n".join(cleaned[:4])

def br(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")

# =====================================================
# CHATGPT-STYLE LOGIC
# =====================================================
GREETINGS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "السلام عليكم"
}

def is_greeting(q: str) -> bool:
    return normalize(q) in GREETINGS

def is_entity_question(q: str) -> bool:
    qn = normalize(q)
    return qn.startswith(("who is", "what is", "who are", "من هو", "من هي", "ما هو", "ما هي"))

def extract_entity(q: str) -> str:
    qn = normalize(q)
    qn = re.sub(r"^(who is|what is|who are)\s+", "", qn).strip()
    qn = re.sub(r"^(من هو|من هي|ما هو|ما هي)\s+", "", qn).strip()
    qn = re.sub(r"[?!.]+$", "", qn).strip()
    return qn

def word_hit(text: str, token: str) -> bool:
    # word boundary-ish check (safe for first-name queries like "faten")
    t = normalize(text)
    tok = normalize(token)
    if not tok:
        return False
    return re.search(rf"(^|[^a-z0-9_]){re.escape(tok)}([^a-z0-9_]|$)", t) is not None

def merge_unique(facts: list[str]) -> str:
    uniq = []
    for f in facts:
        f = (f or "").strip()
        if f and f not in uniq:
            uniq.append(f)
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    return " ".join(uniq)

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# DATA FINGERPRINT (AUTO-REFRESH WHEN TXT CONTENT CHANGES)
# =====================================================
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def data_fingerprint():
    if not os.path.isdir(DATA_DIR):
        return tuple()
    items = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        try:
            items.append((f, file_md5(fp)))
        except Exception:
            continue
    return tuple(sorted(items))

# =====================================================
# BUILD STORES (IMPORTANT FIX)
# - Embed Q + A together (better retrieval)
# - Exact question map first (prevents random wrong answers)
# - Keep docs list for fallback scanning
# =====================================================
@st.cache_resource
def build_assets(_fingerprint):
    if not os.path.isdir(DATA_DIR):
        return None

    emb = get_embeddings()

    docs_by_mode = {"General": [], "Bayut": [], "Dubizzle": []}
    qmap_by_mode = {"General": {}, "Bayut": {}, "Dubizzle": {}}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        bucket = bucket_from_filename(f)

        for q, a in parse_qa_pairs(read_text(fp)):
            nq = normalize(q)
            if nq:
                qmap_by_mode["General"][nq] = a
                qmap_by_mode[bucket][nq] = a

            # 🔥 KEY CHANGE: embed Q + A, not Q only
            content = f"Q: {q}\nA: {a}"
            doc = Document(
                page_content=content,
                metadata={"question": q, "answer": a, "file": f, "bucket": bucket}
            )

            docs_by_mode["General"].append(doc)
            docs_by_mode[bucket].append(doc)

    vs_by_mode = {
        "General": FAISS.from_documents(docs_by_mode["General"], emb) if docs_by_mode["General"] else None,
        "Bayut": FAISS.from_documents(docs_by_mode["Bayut"], emb) if docs_by_mode["Bayut"] else None,
        "Dubizzle": FAISS.from_documents(docs_by_mode["Dubizzle"], emb) if docs_by_mode["Dubizzle"] else None,
    }

    return {"vs": vs_by_mode, "docs": docs_by_mode, "qmap": qmap_by_mode}

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
# ANSWER MODE BUTTONS (Ultra-Fast / Thinking)
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True):
        st.session_state.answer_mode = "Thinking"

# =====================================================
# REFRESH KNOWLEDGE BUTTON (NO “REBOOT”, JUST REINDEX)
# - This updates immediately when you change TXT files.
# =====================================================
refresh_cols = st.columns([6, 2, 6])
with refresh_cols[1]:
    refresh = st.button("Refresh Knowledge", use_container_width=True)

fp = data_fingerprint()

if refresh:
    st.cache_resource.clear()

STORE = build_assets(fp)

def pick_assets():
    if not STORE:
        return None, [], {}
    mode = st.session_state.tool_mode
    return STORE["vs"].get(mode), STORE["docs"].get(mode, []), STORE["qmap"].get(mode, {})

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)
q = st.text_input("", label_visibility="collapsed", key="q_input")

btn_cols = st.columns([1, 1])
ask = btn_cols[0].button("Ask", use_container_width=True)
clear = btn_cols[1].button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []

# =====================================================
# ANSWER (FIXED — CHATGPT STYLE)
# - Greeting: fixed answer
# - Exact Q match: always correct
# - Entity Q: uses BOTH "full question" + "entity" search, aggregates facts
# - Normal Q: semantic search over Q+A
# IMPORTANT: no st.rerun() after ask
# =====================================================
if ask and q:
    vs, docs_mode, qmap_mode = pick_assets()
    qn = normalize(q)
    thinking = st.session_state.answer_mode == "Thinking"

    # 1) Greeting
    if is_greeting(q):
        final = (
            "Hi! 👋 I’m the Bayut & dubizzle Internal AI Assistant. "
            "Ask your question and I’ll answer using the internal knowledge files."
        )

    # 2) Exact match (prevents wrong answers)
    elif qn in qmap_mode:
        final = qmap_mode[qn]

    # 3) Entity question (who/what is X) — aggregate facts
    elif is_entity_question(q):
        if not vs:
            final = "No internal Q&A data found."
        else:
            entity = extract_entity(q)
            if not entity:
                final = "Please specify the name or term you mean."
            else:
                ent = normalize(entity)
                # allow first-name questions to match full names
                variants = {ent}
                parts = [p for p in ent.split(" ") if p]
                if parts:
                    variants.add(parts[0])
                    variants.add(parts[-1])

                # Search using BOTH the full question and the entity token (much better)
                cand = []
                try:
                    cand += vs.similarity_search(q, k=20)
                    cand += vs.similarity_search(entity, k=30)
                except Exception:
                    cand += vs.similarity_search(q, k=12)

                # dedup docs by (file, question)
                seen = set()
                candidates = []
                for d in cand:
                    key = (d.metadata.get("file", ""), normalize(d.metadata.get("question", "")))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(d)

                facts = []
                for d in candidates:
                    qq = d.metadata.get("question", "")
                    aa = d.metadata.get("answer", "")
                    if any(word_hit(qq, v) or word_hit(aa, v) for v in variants):
                        facts.append(aa)

                # fallback scan (fast, capped) if embeddings miss it
                if not facts and docs_mode:
                    hit = 0
                    for d in docs_mode:
                        qq = d.metadata.get("question", "")
                        aa = d.metadata.get("answer", "")
                        if any(word_hit(qq, v) or word_hit(aa, v) for v in variants):
                            facts.append(aa)
                            hit += 1
                            if hit >= 8:
                                break

                merged = merge_unique(facts)

                # If no facts, do NOT return random unrelated answer
                if merged:
                    final = merged
                else:
                    final = f"I couldn’t find information about “{entity}” in the internal knowledge files."

    # 4) Normal question — semantic search (Q + A embedded)
    else:
        if not vs:
            final = "No internal Q&A data found."
        else:
            if thinking:
                with st.spinner("Thinking…"):
                    time.sleep(0.15)

            results = vs.similarity_search(q, k=8 if thinking else 4)
            answers = [r.metadata.get("answer") for r in results if r.metadata.get("answer")]

            if not answers:
                final = "No relevant answer found."
            else:
                final = answers[0] if not thinking else format_thinking_answer(answers[0], answers[1:])

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
