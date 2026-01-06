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
    # strict separation by filename (your rule)
    if "dubizzle" in n:
        return "Dubizzle"
    if "bayut" in n or "mybayut" in n:
        return "Bayut"
    return "General"

def read_text(fp: str) -> str:
    with open(fp, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

# Robust Q/A parsing (supports: Q: / Q13) / Q- , A: / A-)
def parse_qa_pairs(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    pairs: List[Tuple[str, str]] = []

    def is_q(line: str) -> Optional[str]:
        m = re.match(r"^\s*Q\s*\d*\s*[\)\:\-]\s*(.+?)\s*$", line, flags=re.I)
        return m.group(1).strip() if m else None

    def is_a(line: str) -> Optional[str]:
        m = re.match(r"^\s*A\s*[\:\-]\s*(.*)\s*$", line, flags=re.I)
        return m.group(1) if m else None

    curr_q: Optional[str] = None
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

# =====================================================
# KNOWLEDGE SIGNATURE (so Refresh Knowledge works without redeploy)
# =====================================================
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
# BUILD KB (STRICT PER MODE + keep docs for "ChatGPT-style" person answers)
# =====================================================
@st.cache_resource
def build_kb(sig: str):
    if not os.path.isdir(DATA_DIR):
        empty_counts = {"General": 0, "Bayut": 0, "Dubizzle": 0}
        empty_docs = {"General": [], "Bayut": [], "Dubizzle": []}
        return None, None, None, empty_docs, empty_counts, []

    emb = get_embeddings()
    docs_by_bucket: Dict[str, List[Document]] = {"General": [], "Bayut": [], "Dubizzle": []}
    loaded_files: List[str] = []

    for f in sorted(os.listdir(DATA_DIR)):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        bucket = bucket_from_filename(f)
        loaded_files.append(f)

        text = read_text(fp)
        for q, a in parse_qa_pairs(text):
            qn = normalize_space(q)
            an = a.strip()
            content = f"Q: {qn}\nA: {an}"  # IMPORTANT: index Q+Answer together
            docs_by_bucket[bucket].append(
                Document(
                    page_content=content,
                    metadata={
                        "question": qn,
                        "answer": an,
                        "source_file": f,
                        "bucket": bucket,
                    },
                )
            )

    vs_general = FAISS.from_documents(docs_by_bucket["General"], emb) if docs_by_bucket["General"] else None
    vs_bayut = FAISS.from_documents(docs_by_bucket["Bayut"], emb) if docs_by_bucket["Bayut"] else None
    vs_dubizzle = FAISS.from_documents(docs_by_bucket["Dubizzle"], emb) if docs_by_bucket["Dubizzle"] else None
    counts = {k: len(v) for k, v in docs_by_bucket.items()}

    return vs_general, vs_bayut, vs_dubizzle, docs_by_bucket, counts, loaded_files

SIG = kb_signature()
VS_GENERAL, VS_BAYUT, VS_DUBIZZLE, DOCS_BY_BUCKET, KB_COUNTS, LOADED_FILES = build_kb(SIG)

def pick_store():
    return {"General": VS_GENERAL, "Bayut": VS_BAYUT, "Dubizzle": VS_DUBIZZLE}[st.session_state.tool_mode]

def pick_docs():
    return DOCS_BY_BUCKET.get(st.session_state.tool_mode, [])

# =====================================================
# "CHATGPT STYLE" ANSWERS (NO LLM, SMART HEURISTICS)
# =====================================================
def extract_entity(query: str) -> Optional[str]:
    q = (query or "").strip()
    m = re.match(r"^\s*who\s+is\s+(.+?)\s*\?*\s*$", q, flags=re.I)
    if m:
        ent = m.group(1).strip()
        # keep short entities only (prevents weird captures)
        if 1 <= len(ent.split()) <= 5:
            return ent
    return None

def entity_regex(ent: str) -> re.Pattern:
    ent = (ent or "").strip()
    if not ent:
        return re.compile(r"^$")  # no match
    if " " in ent:
        # match words in order
        pat = r"\b" + r"\s+".join(re.escape(w) for w in ent.split()) + r"\b"
        return re.compile(pat, flags=re.I)
    return re.compile(rf"\b{re.escape(ent)}\b", flags=re.I)

def split_names(s: str) -> List[str]:
    s = (s or "").strip()
    s = re.sub(r"[\.]+$", "", s)
    parts = re.split(r"\s*(?:,|&|and)\s*", s, flags=re.I)
    return [p.strip() for p in parts if p.strip()]

def find_app_responsibles(all_docs: List[Document]) -> Optional[str]:
    # Find the Q that states responsibility (from ANY doc in this mode)
    for d in all_docs:
        q = (d.metadata.get("question") or "").lower()
        if "who is responsible for the app" in q or "responsible for the app" in q:
            a = (d.metadata.get("answer") or "").strip()
            return a or None
    return None

def find_full_name(ent: str, all_docs: List[Document]) -> str:
    ent_l = (ent or "").lower().strip()
    # Prefer the "responsible for the app" answer (usually has full names)
    resp = find_app_responsibles(all_docs)
    if resp:
        for nm in split_names(resp):
            if ent_l in nm.lower():
                return nm

    # Otherwise: find a capitalized name that includes the entity
    rx = entity_regex(ent)
    for d in all_docs:
        blob = f"{d.metadata.get('question','')}\n{d.metadata.get('answer','')}"
        if rx.search(blob):
            # look for "Faten Aish" / "Faten ..." (simple + reliable for your data)
            m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", blob)
            if m and ent_l in m.group(1).lower():
                return m.group(1).strip()

    # fallback: title-case the entity
    return ent.strip().title()

def collect_person_docs(ent: str, vs, all_docs: List[Document], thinking: bool) -> List[Document]:
    rx = entity_regex(ent)

    # 1) Direct mentions (fast, accurate)
    direct = []
    for d in all_docs:
        blob = f"{d.metadata.get('question','')}\n{d.metadata.get('answer','')}"
        if rx.search(blob):
            direct.append(d)

    # 2) Semantic hits (to pull "responsible for the app", "Channel Handler", etc.)
    sem = []
    if vs is not None:
        queries = [
            f"who is {ent}",
            ent,
            f"{ent} channel handler",
            "who is responsible for the app",
            "channel handler",
            "responsible for the app",
        ]
        k = 14 if thinking else 10
        for qq in queries:
            try:
                hits = vs.similarity_search(qq, k=k)
                sem.extend(hits)
            except Exception:
                pass

    # Dedup by (file, question)
    out = []
    seen = set()
    for d in direct + sem:
        key = (d.metadata.get("source_file"), d.metadata.get("question"))
        if key not in seen:
            seen.add(key)
            out.append(d)

    return out

def compose_person_answer(ent: str, vs, mode_docs: List[Document], thinking: bool) -> str:
    # IMPORTANT: use ALL docs in this mode for role/responsibility scanning
    all_docs = mode_docs[:]  # already strict per mode
    rel_docs = collect_person_docs(ent, vs, all_docs, thinking)

    full_name = find_full_name(ent, all_docs)
    ent_l = ent.lower().strip()

    # Responsible for app?
    resp = find_app_responsibles(all_docs)
    resp_sentence = ""
    if resp:
        names = split_names(resp)
        # check if person is inside names
        match = None
        for nm in names:
            if ent_l in nm.lower():
                match = nm
                break
        if match:
            others = [n for n in names if n.lower() != match.lower()]
            if others:
                resp_sentence = f"{full_name} is one of the people responsible for the Bayut & dubizzle AI Content Assistant (alongside {', '.join(others)})."
            else:
                resp_sentence = f"{full_name} is responsible for the Bayut & dubizzle AI Content Assistant."

    # Channel Handler / corrections roles?
    blob_all = "\n".join(
        (d.metadata.get("question", "") + "\n" + d.metadata.get("answer", "")) for d in rel_docs
    ).lower()

    role_sentence = ""
    if "channel handler" in blob_all:
        # make it exactly the tone you want
        role_sentence = "She’s also the Channel Handler for corrections and updates—handling coordination, reviewing requests, and tracking progress."
    else:
        # sometimes role is written without the exact phrase
        if ("content (ar)" in blob_all or "arabic content" in blob_all) and "correction" in blob_all:
            role_sentence = "She supports Arabic content corrections and helps coordinate updates."

    # Any extra title if you add it later (example: senior content specialist)
    extra_sentence = ""
    if "senior content specialist" in blob_all:
        extra_sentence = "She’s also a Senior Content Specialist at Bayut."

    # If we still didn't find anything meaningful:
    if not resp_sentence and not role_sentence and not extra_sentence:
        return f"I couldn’t find clear information about “{ent}” in the internal knowledge files for this mode."

    # Build final (clean, not robotic)
    parts = [p for p in [resp_sentence, role_sentence, extra_sentence] if p.strip()]
    return " ".join(parts)

def answer_standard(user_q: str, vs, mode_docs: List[Document], thinking: bool) -> str:
    if vs is None:
        return "No internal Q&A data found for this mode."

    uq_norm = normalize_space(user_q).lower()

    # 1) exact match (best for greetings like "hi", "hello")
    for d in mode_docs:
        if normalize_space(d.metadata.get("question", "")).lower() == uq_norm:
            return (d.metadata.get("answer") or "").strip() or "No relevant answer found."

    # 2) semantic match
    k = 10 if thinking else 6
    try:
        hits = vs.similarity_search_with_score(user_q, k=k)
        docs = [d for d, _s in hits]
    except Exception:
        docs = vs.similarity_search(user_q, k=k)

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

def answer_from_kb(user_q: str, vs, mode_docs: List[Document]) -> str:
    thinking = st.session_state.answer_mode == "Thinking"
    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.12)

    ent = extract_entity(user_q)
    if ent:
        return compose_person_answer(ent, vs, mode_docs, thinking)

    return answer_standard(user_q, vs, mode_docs, thinking)

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

# Keep this so you immediately see if your GitHub /data folder is correct
st.caption(
    f"Loaded Q&A: General {KB_COUNTS.get('General',0)} • Bayut {KB_COUNTS.get('Bayut',0)} • dubizzle {KB_COUNTS.get('Dubizzle',0)}"
)

# =====================================================
# ANSWER MODE + REFRESH KNOWLEDGE (no redeploy needed for TXT changes)
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
# ASK HANDLER (reliable: always appends + reruns)
# =====================================================
if ask_btn:
    if not (q or "").strip():
        st.warning("Type a question first.")
    else:
        try:
            vs = pick_store()
            mode_docs = pick_docs()
            final = answer_from_kb(q, vs, mode_docs)
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
