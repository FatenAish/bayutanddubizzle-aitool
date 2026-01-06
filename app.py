import os
import re
import html
import time
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

      /* Hide Streamlit caption containers (the "Internal AI Assistant" you keep seeing) */
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

      /* QUESTION bubble (same for all modes) */
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

# =====================================================
# HELPERS
# =====================================================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def is_sop_file(name: str) -> bool:
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
    pattern = re.compile(r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|\Z)", re.S | re.I)
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
# SMART Q&A LOGIC HELPERS
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
    return qn.startswith(("who is", "what is", "who are"))

def extract_entity(q: str) -> str:
    qn = normalize(q)
    qn = re.sub(r"^(who is|what is|who are)\s+", "", qn).strip()
    qn = re.sub(r"[?!.]+$", "", qn).strip()
    return qn

def safe_merge_facts(facts: list[str]) -> str:
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
# DATA FINGERPRINT (AUTO-REFRESH WHEN FILES CHANGE)
# =====================================================
def data_fingerprint():
    """
    Makes Streamlit cache rebuild ONLY when data files change
    (so you don't need to "reboot" to pick up updates).
    """
    items = []
    if not os.path.isdir(DATA_DIR):
        return tuple()
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue
        fp = os.path.join(DATA_DIR, f)
        try:
            stt = os.stat(fp)
            items.append((f, int(stt.st_mtime), int(stt.st_size)))
        except Exception:
            continue
    return tuple(sorted(items))

# =====================================================
# BUILD STORES (IMPROVED: EMBED Q + A, PLUS EXACT Q MAP)
# =====================================================
@st.cache_resource
def build_stores(_fingerprint):
    """
    IMPORTANT CHANGE:
    - We embed BOTH question and answer into page_content for better retrieval.
    - We also create an exact normalized question->answer map to prevent wrong matches.
    """
    if not os.path.isdir(DATA_DIR):
        return None

    emb = get_embeddings()

    docs = {"General": [], "Bayut": [], "Dubizzle": []}
    qmap = {"General": {}, "Bayut": {}, "Dubizzle": {}}

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".txt") or is_sop_file(f):
            continue

        fp = os.path.join(DATA_DIR, f)
        b = bucket_from_filename(f)

        for q, a in parse_qa_pairs(read_text(fp)):
            nq = normalize(q)
            if nq:
                # exact lookup maps
                qmap["General"][nq] = a
                qmap[b][nq] = a

            # embed Q + A (this is the biggest accuracy fix)
            content = f"Q: {q}\nA: {a}"
            doc = Document(
                page_content=content,
                metadata={"question": q, "answer": a, "bucket": b, "file": f}
            )

            docs["General"].append(doc)
            docs[b].append(doc)

    vs = {
        "General": FAISS.from_documents(docs["General"], emb) if docs["General"] else None,
        "Bayut": FAISS.from_documents(docs["Bayut"], emb) if docs["Bayut"] else None,
        "Dubizzle": FAISS.from_documents(docs["Dubizzle"], emb) if docs["Dubizzle"] else None,
    }

    return {"vs": vs, "docs": docs, "qmap": qmap}

STORE = build_stores(data_fingerprint())

def pick_assets():
    mode = st.session_state.tool_mode
    if not STORE:
        return None, [], {}
    return STORE["vs"].get(mode), STORE["docs"].get(mode, []), STORE["qmap"].get(mode, {})

# =====================================================
# HEADER (FORCED CENTER + BRAND COLORS)
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

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT (FORCED LABEL: Ask me Anything in bold)
# NOTE: No st.rerun() on Ask anymore.
# =====================================================
st.markdown("<div class='ask-label'>Ask me Anything</div>", unsafe_allow_html=True)

# Using a form prevents extra reruns from typing; it only reruns on submit.
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("", label_visibility="collapsed", key="q_input")
    btn_cols = st.columns([1, 1])
    ask = btn_cols[0].form_submit_button("Ask", use_container_width=True)
    clear = btn_cols[1].form_submit_button("Clear chat", use_container_width=True)

# Clear (NO st.rerun() needed)
if clear:
    st.session_state.chat[st.session_state.tool_mode] = []

# =====================================================
# ANSWER (FIXED LOGIC: no random wrong answers)
# =====================================================
if ask and q:
    vs, docs_mode, qmap_mode = pick_assets()

    qn = normalize(q)
    thinking = st.session_state.answer_mode == "Thinking"

    # 1) Greeting -> fixed response
    if is_greeting(q):
        final = (
            "Hi! 👋 I’m the Bayut & dubizzle Internal AI Assistant. "
            "Ask your question and I’ll answer using the internal knowledge files."
        )

    # 2) Exact question match (prevents wrong FAISS hits)
    elif qn in qmap_mode:
        final = qmap_mode[qn]

    # 3) Entity question -> ONLY answer if we actually find facts about that entity
    elif is_entity_question(q):
        if not vs:
            final = "No internal Q&A data found."
        else:
            entity = extract_entity(q)
            if not entity:
                final = "Please specify the name or term you mean."
            else:
                # pull candidates by semantic search
                candidates = vs.similarity_search(entity, k=25)
                facts = []
                ent = normalize(entity)

                for d in candidates:
                    ans = d.metadata.get("answer", "")
                    qq = d.metadata.get("question", "")
                    if ent and (ent in normalize(ans) or ent in normalize(qq)):
                        facts.append(ans)

                # If still empty, do a light scan of the mode docs (fast, capped)
                if not facts and docs_mode:
                    hit = 0
                    for d in docs_mode:
                        ans = d.metadata.get("answer", "")
                        qq = d.metadata.get("question", "")
                        if ent and (ent in normalize(ans) or ent in normalize(qq)):
                            facts.append(ans)
                            hit += 1
                            if hit >= 8:
                                break

                merged = safe_merge_facts(facts)

                # IMPORTANT: If no facts, do NOT fallback to random answers.
                if merged:
                    final = merged
                else:
                    final = f"I couldn’t find information about “{entity}” in the internal knowledge files."

    # 4) Normal question -> semantic search, but with better embeddings (Q+A)
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

    # Save in chat (NO st.rerun())
    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})

# =====================================================
# CHAT HISTORY (ANSWER bubble changes by mode: gray/green/red)
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
