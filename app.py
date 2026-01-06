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
# ANSWER LOGIC (CHATGPT-STYLE WITHOUT LLM)
# =====================================================

def _clean(a: str) -> str:
    a = (a or "").strip()
    a = re.sub(r"\n{3,}", "\n\n", a)
    return a

def _split_lines(text: str) -> list[str]:
    lines = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
    return lines

def _title_case_name(name: str) -> str:
    # keep acronyms and Arabic as-is
    if re.search(r"[\u0600-\u06FF]", name):
        return name
    return " ".join([w.capitalize() if w.islower() else w for w in name.split()])

def _extract_entity(query: str) -> str | None:
    q = (query or "").strip()
    m = re.match(r"^\s*who\s+is\s+(.+?)\s*\?*\s*$", q, flags=re.I)
    if m:
        ent = m.group(1).strip()
        # keep short entities only
        if 1 <= len(ent.split()) <= 5:
            return ent
    # Arabic "من هو/من هي"
    m2 = re.match(r"^\s*(من هو|من هي)\s+(.+?)\s*$", q, flags=re.I)
    if m2:
        ent = m2.group(2).strip()
        if 1 <= len(ent.split()) <= 5:
            return ent
    return None

def _collect_entity_facts(entity: str, docs: list[Document], thinking: bool) -> dict:
    """
    Pull ONLY the useful facts about the entity from different answers,
    then we will write a nice response.
    """
    ent_l = entity.lower().strip()
    facts = {
        "canonical_name": None,
        "responsible_with": None,     # list of names
        "roles": [],                 # key roles lines
        "extra": [],                 # other lines mentioning entity
    }

    role_keywords = [
        "responsible", "owner", "managed",
        "channel handler", "coordination",
        "lead", "specialist", "manager",
        "content", "operations"
    ]

    for d in docs:
        ans = _clean(d.metadata.get("answer", ""))

        # Grab lines that mention the entity
        lines = _split_lines(ans)
        hit_lines = [ln for ln in lines if ent_l in ln.lower()]

        # If entity appears, set canonical name if we can
        if hit_lines and not facts["canonical_name"]:
            # try to capture "Faten Aish" style from the line
            # fallback to entity itself
            facts["canonical_name"] = hit_lines[0]
            # better: extract first two words that look like a name
            m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", hit_lines[0])
            if m:
                facts["canonical_name"] = m.group(1).strip()

        # RESPONSIBLE pattern: "X and Y"
        if "responsible for the app" in (d.metadata.get("question", "").lower()):
            # Answer likely is "Faten Aish and Sarah Al Nawah."
            names = [n.strip().strip(".") for n in re.split(r"\band\b|&|،", ans, flags=re.I) if n.strip()]
            if names:
                facts["responsible_with"] = names

        # Roles: pick lines mentioning entity that also contain role keywords
        for ln in hit_lines:
            ln_l = ln.lower()
            if any(k in ln_l for k in role_keywords):
                if ln not in facts["roles"]:
                    facts["roles"].append(ln)
            else:
                if ln not in facts["extra"]:
                    facts["extra"].append(ln)

    # Trim
    max_roles = 4 if thinking else 2
    max_extra = 4 if thinking else 1
    facts["roles"] = facts["roles"][:max_roles]
    facts["extra"] = facts["extra"][:max_extra]
    return facts

def _render_entity_answer(entity: str, facts: dict) -> str:
    """
    Write a friendly, natural response (ChatGPT-ish) from the extracted facts.
    """
    name = facts.get("canonical_name") or _title_case_name(entity)

    parts = []

    # Responsible for the app
    rw = facts.get("responsible_with")
    if rw:
        # If entity is one of them, phrase nicely
        if any(entity.lower() in n.lower() for n in rw):
            # Show both names cleanly
            if len(rw) >= 2:
                parts.append(f"{rw[0]} is one of the people responsible for this app, together with {rw[1]}.")
            else:
                parts.append(f"{rw[0]} is responsible for this app.")
        else:
            # still useful info
            if len(rw) >= 2:
                parts.append(f"This app is managed by {rw[0]} and {rw[1]}.")

    # Roles (Channel Handler etc.)
    roles = facts.get("roles") or []
    if roles:
        # Convert role lines into nice sentences (light cleanup)
        role_sentences = []
        for r in roles:
            rr = r.strip().rstrip(".")
            rr = rr.replace("—", "-")
            # common line: "Faten Aish is the Channel Handler. They:"
            rr = rr.replace("They:", "").strip()
            # If line is "Content (AR): Faten Aish — Arabic Content Corrections / Channel Handler"
            if "content (ar)" in rr.lower():
                role_sentences.append(f"She also acts as the Channel Handler for Arabic content corrections.")
            elif "channel handler" in rr.lower():
                role_sentences.append(f"She also acts as the Channel Handler for this workflow.")
            else:
                # keep as supporting fact
                role_sentences.append(rr)

        # Dedup and add
        cleaned = []
        for s in role_sentences:
            s = s.strip()
            if s and s not in cleaned:
                cleaned.append(s)
        parts.extend(cleaned[:2])

    # Extra facts (if any)
    extra = facts.get("extra") or []
    # only add extras if they add value
    useful_extra = []
    for e in extra:
        e = e.strip().rstrip(".")
        if e and e not in useful_extra and "content (ar)" not in e.lower():
            useful_extra.append(e)

    if useful_extra:
        # Keep it simple — one extra line
        parts.append(useful_extra[0] + ".")

    # If we still have nothing, fallback
    if not parts:
        return f"I found mentions of {name} in the knowledge files, but not enough to give a clear profile. Try asking with more context (team/process)."

    # Make it read well
    out = " ".join(parts)

    # Replace repeated "She" if entity isn't clearly female/unknown
    # (skip – you can keep it as-is if your team knows the person)
    return out

def answer_from_store(user_q: str, vs):
    if vs is None:
        return "No internal Q&A data found for this mode."

    thinking = st.session_state.answer_mode == "Thinking"
    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.12)

    # Fetch candidates using both the full question and (if entity query) the entity alone
    ent = _extract_entity(user_q)

    hits = []
    try:
        hits.extend(vs.similarity_search_with_score(user_q, k=12 if thinking else 8))
        if ent:
            hits.extend(vs.similarity_search_with_score(ent, k=12 if thinking else 8))
    except Exception:
        # fallback if score search isn't available
        docs = vs.similarity_search(user_q, k=12 if thinking else 8)
        hits.extend([(d, 1.0) for d in docs])
        if ent:
            docs2 = vs.similarity_search(ent, k=12 if thinking else 8)
            hits.extend([(d, 1.0) for d in docs2])

    if not hits:
        return "No relevant answer found."

    # Dedup docs while preserving order
    seen = set()
    docs_sorted = []
    for d, _s in hits:
        key = (d.metadata.get("source_file"), d.metadata.get("question"))
        if key not in seen:
            seen.add(key)
            docs_sorted.append(d)

    # ENTITY MODE: "who is X" → compose profile answer from multiple facts
    if ent:
        facts = _collect_entity_facts(ent, docs_sorted, thinking=thinking)
        return _render_entity_answer(ent, facts)

    # NORMAL MODE: return top answer but make it nicer (avoid massive blocks)
    answers = [(_clean(d.metadata.get("answer", "")), d) for d in docs_sorted]
    answers = [(a, d) for a, d in answers if a]

    if not answers:
        return "No relevant answer found."

    top = answers[0][0]

    # If the answer is very long, show only the most useful part
    if len(top) > 600 and st.session_state.answer_mode != "Thinking":
        # keep first 2 paragraphs
        paras = [p.strip() for p in top.split("\n\n") if p.strip()]
        top = "\n\n".join(paras[:2])

    if st.session_state.answer_mode != "Thinking":
        return top

    # Thinking mode: show up to 3 distinct helpful answers
    extras = [a for a, _d in answers[1:]]
    return format_thinking_answer(top, extras)

# ---- Run answer ----
if ask and q:
    final = answer_from_store(q, pick_store())
    st.session_state.chat[st.session_state.tool_mode].append({"q": q, "a": final})
    st.rerun()
