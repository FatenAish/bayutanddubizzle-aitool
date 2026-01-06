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
      /* Force container width */
      section.main > div.block-container{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
      }

      /* Hide Streamlit caption containers */
      div[data-testid="stCaptionContainer"]{ display:none !important; }
      div[data-testid="stHeader"]{ background: transparent !important; }
      header{ background: transparent !important; }

      /* Center ALL h1 inside the app */
      section.main h1{
        text-align:center !important;
        margin-bottom: 6px !important;
      }

      /* Brand header block */
      #brand-header{
        text-align:center !important;
        margin: 0 0 10px 0 !important;
      }
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

      /* Buttons */
      div.stButton > button{
        border-radius: 12px !important;
        font-weight: 600 !important;
      }

      /* Custom label for input */
      .ask-label{
        font-weight: 800 !important;
        font-size: 1rem !important;
        margin: 12px 0 6px 0 !important;
      }

      /* QUESTION bubble */
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

      /* ANSWER bubbles per mode */
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

      /* Center the "Assistant" heading */
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
st.session_state.setdefault("kb_bust", 0)  # manual refresh token if needed

# =====================================================
# HELPERS
# =====================================================
def is_sop_file(name: str) -> bool:
    # keep SOPs excluded from Q&A KB (your preference)
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "dubizzle" in n:
        return "Dubizzle"
    # MyBayut is Bayut brand context
    if "bayut" in n or "mybayut" in n:
        return "Bayut"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def br(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def looks_like_q_line(line: str) -> str | None:
    """
    Supports:
      Q: ...
      Q) ...
      Q5) ...
      Q13) ...
      Q- ...
    Returns extracted question text or None.
    """
    m = re.match(r"^\s*Q\s*\d*\s*[\)\:\-]\s*(.+?)\s*$", line, flags=re.I)
    if m:
        return m.group(1).strip()
    return None

def looks_like_a_line(line: str) -> str | None:
    """
    Supports:
      A: ...
      A- ...
    Returns extracted answer text (can be empty) or None.
    """
    m = re.match(r"^\s*A\s*[\:\-]\s*(.*)\s*$", line, flags=re.I)
    if m:
        return m.group(1)
    return None

def next_non_empty(lines, start_idx: int) -> str:
    for j in range(start_idx, len(lines)):
        t = lines[j].strip()
        if t:
            return t
    return ""

def parse_qa_pairs(text: str):
    """
    Robust parser that handles:
    - Q: / Q) / Q13) styles
    - multi-line answers (including A: empty then content)
    - question lines without Q prefix IF they end with '?' and next non-empty line starts with 'A:'
    """
    lines = text.splitlines()
    pairs = []

    curr_q = None
    curr_a_lines = []
    in_answer = False

    def flush():
        nonlocal curr_q, curr_a_lines, in_answer
        if curr_q is not None:
            a = "\n".join(curr_a_lines).strip()
            q = curr_q.strip()
            if q and a:
                pairs.append((q, a))
        curr_q = None
        curr_a_lines = []
        in_answer = False

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Detect a new Q line
        q_txt = looks_like_q_line(line)
        if q_txt:
            flush()
            curr_q = q_txt
            in_answer = False
            i += 1
            continue

        # Detect a question without "Q:" if it ends with "?" and next line is "A:"
        if line.endswith("?"):
            nxt = next_non_empty(lines, i + 1)
            if looks_like_a_line(nxt) is not None:
                flush()
                curr_q = line
                in_answer = False
                i += 1
                continue

        # Detect A line (start answer)
        a_txt = looks_like_a_line(line)
        if a_txt is not None and curr_q is not None:
            in_answer = True
            curr_a_lines.append(a_txt.strip())
            i += 1
            continue

        # Collect answer lines until next question
        if curr_q is not None and in_answer:
            # Keep content (even if it's like "- item")
            if line != "":
                curr_a_lines.append(raw.rstrip())
        i += 1

    flush()
    return pairs

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = [primary] + extras
    cleaned = []
    for x in out:
        x = (x or "").strip()
        if x and x not in cleaned:
            cleaned.append(x)
    return "\n\n".join(cleaned[:4])

def extract_entity(query: str) -> str | None:
    """
    Simple "who is X" detector.
    Returns X (lower) or None.
    """
    q = query.strip().lower()
    m = re.match(r"^\s*who\s+is\s+(.+?)\s*\?*\s*$", q)
    if m:
        ent = m.group(1).strip()
        # keep it short (avoid "who is responsible..." etc)
        if 1 <= len(ent.split()) <= 4:
            return ent
    return None

def extract_lines_with_term(text: str, term: str, max_lines: int = 3) -> list[str]:
    term_l = term.lower()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    hits = [ln for ln in lines if term_l in ln.lower()]
    return hits[:max_lines]

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def compute_kb_signature() -> str:
    """
    Makes cache auto-refresh when files change (GitHub deploy, updated data, etc).
    """
    if not os.path.isdir(DATA_DIR):
        return "no-data-dir"

    parts = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.lower().endswith(".txt"):
            continue
        if is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        try:
            stt = os.stat(fp)
            parts.append(f"{f}:{stt.st_size}:{int(stt.st_mtime)}")
        except Exception:
            parts.append(f"{f}:missing")
    raw = "|".join(parts) + f"|bust:{st.session_state.kb_bust}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

# =====================================================
# BUILD STORES (STRICT MODE SEPARATION + SEARCH IN Q&A TOGETHER)
# =====================================================
@st.cache_resource
def build_stores(kb_sig: str):
    """
    - General mode loads ONLY General files
    - Bayut mode loads ONLY Bayut/MyBayut files
    - Dubizzle mode loads ONLY Dubizzle files
    - Each QA pair is indexed with BOTH Q + A so entity queries work.
    """
    if not os.path.isdir(DATA_DIR):
        return None, None, None

    emb = get_embeddings()
    docs = {"General": [], "Bayut": [], "Dubizzle": []}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        text = read_text(fp)
        pairs = parse_qa_pairs(text)

        bucket = bucket_from_filename(f)

        for q, a in pairs:
            q_clean = normalize_space(q)
            a_clean = a.strip()

            # IMPORTANT: index BOTH Q + A (so "who is faten" can match content where Faten appears in the answer)
            page = f"Q: {q_clean}\nA: {a_clean}"

            doc = Document(
                page_content=page,
                metadata={
                    "answer": a_clean,
                    "question": q_clean,
                    "source_file": f,
                    "bucket": bucket,
                },
            )
            docs[bucket].append(doc)

    vs_general = FAISS.from_documents(docs["General"], emb) if docs["General"] else None
    vs_bayut = FAISS.from_documents(docs["Bayut"], emb) if docs["Bayut"] else None
    vs_dubizzle = FAISS.from_documents(docs["Dubizzle"], emb) if docs["Dubizzle"] else None
    return vs_general, vs_bayut, vs_dubizzle

KB_SIG = compute_kb_signature()
VS_GENERAL, VS_BAYUT, VS_DUBIZZLE = build_stores(KB_SIG)

def pick_store():
    return {"General": VS_GENERAL, "Bayut": VS_BAYUT, "Dubizzle": VS_DUBIZZLE}[st.session_state.tool_mode]

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
# ANSWER MODE BUTTONS + REFRESH KNOWLEDGE
# =====================================================
mode_cols = st.columns([4, 2, 2, 2, 4])
with mode_cols[1]:
    if st.button("Ultra-Fast", use_container_width=True):
        st.session_state.answer_mode = "Ultra-Fast"
with mode_cols[2]:
    if st.button("Thinking", use_container_width=True):
        st.session_state.answer_mode = "Thinking"
with mode_cols[3]:
    if st.button("Refresh Knowledge", use_container_width=True):
        # no reboot needed: bust cache key then rerun
        st.session_state.kb_bust += 1
        st.rerun()

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
    st.rerun()

# =====================================================
# ANSWER LOGIC (CHATGPT-STYLE: extract the relevant fact, not a random block)
# =====================================================
def answer_from_store(user_q: str, vs):
    if vs is None:
        return "No internal Q&A data found for this mode."

    thinking = st.session_state.answer_mode == "Thinking"
    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.15)

    # use scores so we can filter weak matches
    results = vs.similarity_search_with_score(user_q, k=10 if thinking else 6)

    if not results:
        return "No relevant answer found."

    # optional: filter weak matches (tune if needed)
    # NOTE: FAISS score meaning depends on distance; keep it permissive.
    docs_sorted = [d for (d, _s) in results]

    # Entity-style questions: "who is X"
    ent = extract_entity(user_q)
    if ent:
        facts = []
        for d in docs_sorted:
            ans = d.metadata.get("answer", "")
            hits = extract_lines_with_term(ans, ent, max_lines=2)
            facts.extend(hits)

        # If we found entity facts, return them cleanly (dedup)
        if facts:
            out = []
            for f in facts:
                if f not in out:
                    out.append(f)
            return "\n".join(out[:4])

        # fallback: if no direct entity line, return best normal answer
        best = docs_sorted[0].metadata.get("answer", "").strip()
        return best or "No relevant answer found."

    # Normal Q&A: return top answer, and in Thinking mode add a couple supporting ones
    answers = [d.metadata.get("answer", "").strip() for d in docs_sorted]
    answers = [a for a in answers if a]

    if not answers:
        return "No relevant answer found."

    if not thinking:
        return answers[0]

    return format_thinking_answer(answers[0], answers[1:])

if ask and q:
    final = answer_from_store(q, pick_store())
    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})
    st.rerun()

# =====================================================
# CHAT HISTORY
# =====================================================
answer_class = {"General": "a-general", "Bayut": "a-bayut", "Dubizzle": "a-dubizzle"}[st.session_state.tool_mode]

for item in reversed(st.session_state.chat[st.session_state.tool_mode]):
    st.markdown(f"<div class='q-bubble'>{br(item['q'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='a-bubble {answer_class}'>{br(item['a'])}</div>", unsafe_allow_html=True)
    st.markdown("---")
