import os
import re
import html
import time
import hashlib
from typing import List, Tuple, Optional, Dict

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
# HARD-FORCE STYLES (KEEP YOUR DESIGN)
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

      section.main h1{ text-align:center !important; margin-bottom: 6px !important; }

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
st.session_state.setdefault("kb_bust", 0)

# =====================================================
# UTIL
# =====================================================
def br(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_sop_file(name: str) -> bool:
    return "sop" in name.lower()

def bucket_from_filename(name: str) -> str:
    n = name.lower()
    if "dubizzle" in n:
        return "Dubizzle"
    if "bayut" in n or "mybayut" in n:
        return "Bayut"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

# ---- Robust Q/A parsing (supports Q13), Q), Q:, etc.) ----
def parse_qa_pairs(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    pairs: List[Tuple[str, str]] = []

    def is_q(line: str) -> Optional[str]:
        m = re.match(r"^\s*Q\s*\d*\s*[\)\:\-]\s*(.+?)\s*$", line, flags=re.I)
        return m.group(1).strip() if m else None

    def is_a(line: str) -> Optional[str]:
        m = re.match(r"^\s*A\s*[\:\-]\s*(.*)\s*$", line, flags=re.I)
        return m.group(1) if m else None

    curr_q = None
    curr_a: List[str] = []
    in_a = False

    def flush():
        nonlocal curr_q, curr_a, in_a
        if curr_q:
            a = "\n".join(curr_a).strip()
            q = curr_q.strip()
            if q and a:
                pairs.append((q, a))
        curr_q = None
        curr_a = []
        in_a = False

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        qtxt = is_q(line)
        if qtxt:
            flush()
            curr_q = qtxt
            in_a = False
            i += 1
            continue

        atxt = is_a(line)
        if atxt is not None and curr_q is not None:
            in_a = True
            curr_a.append(atxt.strip())
            i += 1
            continue

        if curr_q is not None and in_a:
            if line:
                curr_a.append(raw.rstrip())

        i += 1

    flush()
    return pairs

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def kb_signature() -> str:
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
# BUILD STORES (STRICT PER MODE) + KEEP ALL DOCS
# =====================================================
@st.cache_resource
def build_stores(sig: str):
    if not os.path.isdir(DATA_DIR):
        empty_counts = {"General": 0, "Bayut": 0, "Dubizzle": 0}
        empty_docs = {"General": [], "Bayut": [], "Dubizzle": []}
        return None, None, None, empty_counts, empty_docs

    emb = get_embeddings()
    docs_by_mode: Dict[str, List[Document]] = {"General": [], "Bayut": [], "Dubizzle": []}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        bucket = bucket_from_filename(f)
        text = read_text(fp)

        for q, a in parse_qa_pairs(text):
            q = normalize_space(q)
            a = (a or "").strip()

            content = f"Q: {q}\nA: {a}"
            docs_by_mode[bucket].append(
                Document(
                    page_content=content,
                    metadata={
                        "question": q,
                        "answer": a,
                        "source_file": f,
                        "bucket": bucket,
                    },
                )
            )

    vs_general = FAISS.from_documents(docs_by_mode["General"], emb) if docs_by_mode["General"] else None
    vs_bayut = FAISS.from_documents(docs_by_mode["Bayut"], emb) if docs_by_mode["Bayut"] else None
    vs_dubizzle = FAISS.from_documents(docs_by_mode["Dubizzle"], emb) if docs_by_mode["Dubizzle"] else None

    counts = {k: len(v) for k, v in docs_by_mode.items()}
    return vs_general, vs_bayut, vs_dubizzle, counts, docs_by_mode

SIG = kb_signature()
VS_GENERAL, VS_BAYUT, VS_DUBIZZLE, KB_COUNTS, DOCS_BY_MODE = build_stores(SIG)

def pick_store():
    return {"General": VS_GENERAL, "Bayut": VS_BAYUT, "Dubizzle": VS_DUBIZZLE}[st.session_state.tool_mode]

# =====================================================
# PERSON ANSWERS (CHATGPT-STYLE, NO FAISS DEPENDENCY)
# =====================================================
def extract_entity(query: str) -> Optional[str]:
    q = (query or "").strip()
    m = re.match(r"^\s*who\s+is\s+(.+?)\s*\?*\s*$", q, flags=re.I)
    if m:
        ent = m.group(1).strip()
        if 1 <= len(ent.split()) <= 6:
            return ent
    return None

def all_docs_flat() -> List[Document]:
    out = []
    for k in ("General", "Bayut", "Dubizzle"):
        out.extend(DOCS_BY_MODE.get(k, []) or [])
    return out

def find_app_owners_answer(docs: List[Document]) -> Optional[str]:
    # One source of truth: the Q/A "Who is responsible for the app?"
    for d in docs:
        q = (d.metadata.get("question") or "").lower()
        a = (d.metadata.get("answer") or "").strip()
        if not a:
            continue
        if "responsible for the app" in q or "who is responsible for the app" in q:
            return a.rstrip(".")
    return None

def extract_role_from_line(line: str, ent_l: str) -> Optional[str]:
    # Clean prefixes like "Content (AR):"
    s = line.strip()
    if ":" in s:
        left, right = s.split(":", 1)
        # keep if name is on right
        s = right.strip()

    # Now try split on em dash
    if "—" in s:
        left, role = s.split("—", 1)
        if ent_l in left.lower():
            return role.strip()
    # Try split on " - "
    if " - " in s:
        left, role = s.split(" - ", 1)
        if ent_l in left.lower():
            return role.strip()

    return None

def compose_person_answer(ent: str) -> str:
    ent_l = ent.lower().strip()

    docs = all_docs_flat()  # IMPORTANT: scan everything
    owners = find_app_owners_answer(docs)

    # Find best full name (if available)
    full_name = None
    roles = []
    channel_handler = False

    for d in docs:
        a = (d.metadata.get("answer") or "")
        if not a:
            continue

        # look for lines with the entity
        for ln in a.splitlines():
            lnl = ln.lower()
            if ent_l in lnl:
                # capture a full name like "Faten Aish"
                m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", ln)
                if m:
                    full_name = full_name or m.group(1).strip()

                role = extract_role_from_line(ln, ent_l)
                if role:
                    roles.append(role)

                if "channel handler" in lnl:
                    channel_handler = True

    # Dedup roles
    uniq_roles = []
    for r in roles:
        r = r.strip().rstrip(".")
        if r and r not in uniq_roles:
            uniq_roles.append(r)

    name = full_name or ent.strip().title()

    parts = []

    # Always include owners if relevant to this person
    if owners:
        owners_l = owners.lower()
        if ent_l in owners_l or (ent_l in ("faten", "sarah") and ("faten" in owners_l or "sarah" in owners_l)):
            # Make it human
            if "sarah" in owners_l and "faten" in owners_l:
                parts.append(f"{name} is one of the people responsible for the app, together with Sarah Al Nawah.")
            else:
                parts.append(f"{name} is responsible for the app.")

    # Add key role(s)
    # If they added “Senior Content Specialist…”, it will appear here naturally.
    picked = []
    for r in uniq_roles:
        rl = r.lower()
        if "senior" in rl or "specialist" in rl:
            picked.append(r)
        elif "arabic" in rl or "content corrections" in rl:
            picked.append(r)
        elif "channel handler" in rl:
            picked.append(r)

    # If nothing matched the filters, just take first role
    if not picked and uniq_roles:
        picked = uniq_roles[:1]

    # Turn roles into natural sentence(s)
    if picked:
        # Example: "Arabic Content Corrections / Channel Handler"
        # Keep it clean:
        role_text = picked[0].replace("/", " and ").strip()
        parts.append(f"She works on {role_text}.")

    # Channel handler flag (if not already covered)
    if channel_handler and not any("channel handler" in p.lower() for p in parts):
        parts.append("She also acts as the Channel Handler for corrections and updates (content coordination).")

    if not parts:
        return f"I couldn’t find clear information about “{ent}” in the internal knowledge files."

    # Final cleanup (avoid repeated "She")
    out = " ".join(parts)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# =====================================================
# ANSWER FROM STORE (NORMAL QUESTIONS)
# =====================================================
def answer_from_store(user_q: str, vs):
    if vs is None:
        return "No internal Q&A data found for this mode."

    thinking = st.session_state.answer_mode == "Thinking"
    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.10)

    ent = extract_entity(user_q)
    if ent:
        return compose_person_answer(ent)

    # Normal Q/A retrieval
    docs = []
    try:
        hits = vs.similarity_search_with_score(user_q, k=10 if thinking else 6)
        docs = [d for d, _s in hits]
    except Exception:
        docs = vs.similarity_search(user_q, k=10 if thinking else 6)

    if not docs:
        return "No relevant answer found."

    answers = []
    for d in docs:
        a = (d.metadata.get("answer") or "").strip()
        if a and a not in answers:
            answers.append(a)

    if not answers:
        return "No relevant answer found."

    if not thinking:
        return answers[0]

    return "\n\n".join(answers[:3])

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
# ANSWER MODE + REFRESH KNOWLEDGE
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
        st.session_state.kb_bust += 1
        st.rerun()

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)
q = st.text_input("", label_visibility="collapsed", key="q_input")

btn_cols = st.columns([1, 1])
ask_btn = btn_cols[0].button("Ask", use_container_width=True)
clear_btn = btn_cols[1].button("Clear chat", use_container_width=True)

if clear_btn:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ASK HANDLER
# =====================================================
if ask_btn:
    if not (q or "").strip():
        st.warning("Type a question first.")
    else:
        try:
            vs = pick_store()
            final = answer_from_store(q, vs)
            st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})
            st.rerun()
        except Exception as e:
            st.error("Something broke while answering. Here is the real error:")
            st.exception(e)

# =====================================================
# CHAT HISTORY
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
