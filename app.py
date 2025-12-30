import os
import re
import html
import time
import base64
import hashlib
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================
# PAGE CONFIG (must be first Streamlit call)
# =====================================================
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =====================================================
# BACKGROUND (FULL WEBSITE)
# ✅ You uploaded background.png under /data, so we prioritize it
# =====================================================
def _find_background_image():
    preferred_exact = [
        # ✅ YOUR CURRENT PLACE (data)
        os.path.join(DATA_DIR, "background.png"),
        os.path.join(DATA_DIR, "background.jpg"),
        os.path.join(DATA_DIR, "background.jpeg"),

        # assets (optional)
        os.path.join(ASSETS_DIR, "background.png"),
        os.path.join(ASSETS_DIR, "background.jpg"),
        os.path.join(ASSETS_DIR, "background.jpeg"),

        # root (optional)
        os.path.join(BASE_DIR, "background.png"),
        os.path.join(BASE_DIR, "background.jpg"),
        os.path.join(BASE_DIR, "background.jpeg"),
    ]

    for p in preferred_exact:
        if p and os.path.isfile(p):
            return p

    # fallback: any image in assets
    if os.path.isdir(ASSETS_DIR):
        imgs = [x for x in os.listdir(ASSETS_DIR) if x.lower().endswith((".png", ".jpg", ".jpeg"))]
        if imgs:
            return os.path.join(ASSETS_DIR, sorted(imgs)[0])

    # fallback: any image in root
    imgs_root = [x for x in os.listdir(BASE_DIR) if x.lower().endswith((".png", ".jpg", ".jpeg"))]
    if imgs_root:
        return os.path.join(BASE_DIR, sorted(imgs_root)[0])

    # fallback: any image in data
    if os.path.isdir(DATA_DIR):
        imgs_data = [x for x in os.listdir(DATA_DIR) if x.lower().endswith((".png", ".jpg", ".jpeg"))]
        if imgs_data:
            # IMPORTANT: if you have other images in /data, this might pick the wrong one.
            # That's why we always recommend naming it exactly background.png above.
            return os.path.join(DATA_DIR, sorted(imgs_data)[0])

    return None

@st.cache_data(show_spinner=False)
def _img_to_data_uri(path: str, mtime: float):
    # mtime is only used to bust cache when image changes
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

_BG_PATH = _find_background_image()
_BG_URI = _img_to_data_uri(_BG_PATH, os.path.getmtime(_BG_PATH)) if _BG_PATH else None

# =====================================================
# CSS (FULL PAGE BACKGROUND)
# =====================================================
if _BG_URI:
    bg_value = f"url('{_BG_URI}')"
else:
    bg_value = "linear-gradient(180deg,#0e5b76 0%, #0a3d4f 100%)"

st.markdown(
    f"""
    <style>
      :root {{
        --app-bg: {bg_value};
      }}

      /* FULL WEBSITE BACKGROUND (ALL LAYERS) */
      html, body {{
        height: 100%;
        min-height: 100%;
        background: var(--app-bg) !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
      }}

      .stApp {{
        min-height: 100vh !important;
        background: var(--app-bg) !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
      }}

      [data-testid="stAppViewContainer"] {{
        min-height: 100vh !important;
        background: var(--app-bg) !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
      }}

      [data-testid="stAppViewContainer"] > .main {{
        min-height: 100vh !important;
        background: transparent !important;
      }}

      [data-testid="stHeader"] {{
        background: transparent !important;
      }}

      /* MAIN CONTENT GLASS CARD */
      section.main > div.block-container {{
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;

        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        border-radius: 22px !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.18) !important;
      }}

      .center {{ text-align:center; }}

      .q-bubble {{
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 85%;
        width: fit-content;
        font-weight: 600;
        margin: 10px 0 8px;
        border: 1px solid rgba(0,0,0,0.06);
      }}
      .q-general {{ background:#f2f2f2; }}
      .q-bayut {{ background:#e6f4ef; }}
      .q-dubizzle {{ background:#fdeaea; }}

      .answer {{
        margin-left: 6px;
        margin-bottom: 14px;
        line-height: 1.6;
      }}

      div.stButton > button {{ border-radius: 10px; }}

      /* Smaller Ultra-Fast / Thinking buttons */
      .small-btn div.stButton > button {{
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        font-size: 0.95rem !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# ACCESS CODE GATE (FIRST SCREEN)
# =====================================================
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
REQUIRE_CODE = os.getenv("REQUIRE_CODE", "0").strip() == "1"

def _get_qp():
    try:
        return st.query_params
    except Exception:
        return st.experimental_get_query_params()

def _set_qp(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

if REQUIRE_CODE and ACCESS_CODE:
    st.session_state.setdefault("unlocked", False)

    qp = _get_qp()
    qp_code = None
    try:
        qp_code = qp.get("code")
        if isinstance(qp_code, list):
            qp_code = qp_code[0] if qp_code else None
    except Exception:
        qp_code = None

    if qp_code and qp_code == ACCESS_CODE:
        st.session_state["unlocked"] = True
        try:
            _set_qp()
        except Exception:
            pass

    if not st.session_state["unlocked"]:
        st.markdown(
            """
            <style>
              section.main > div.block-container{
                max-width: 520px !important;
                padding-top: 6rem !important;
              }
              .gate-wrap{ text-align:center; }
              .gate-title{ font-size: 34px; font-weight: 900; margin-bottom: 6px; }
              .gate-sub{ color:#666; margin-bottom: 22px; }
            </style>

            <div class="gate-wrap">
              <div class="gate-title">
                <span style="color:#0E8A6D;">Bayut</span> &
                <span style="color:#D71920;">Dubizzle</span> AI Assistant
              </div>
              <div class="gate-sub">Internal AI Assistant – Access Required</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        code = st.text_input(
            "Access code",
            type="password",
            placeholder="Enter access code",
            label_visibility="collapsed"
        )

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            unlock = st.button("Unlock", use_container_width=True)

        if unlock:
            if code == ACCESS_CODE:
                st.session_state["unlocked"] = True
                st.rerun()
            else:
                st.error("Wrong access code")

        st.stop()

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("tool_mode", "General")
st.session_state.setdefault("answer_mode", "Ultra-Fast")
st.session_state.setdefault("chat", {"General": [], "Bayut": [], "Dubizzle": []})

# =====================================================
# HELPERS
# =====================================================
def read_text(fp: str) -> str:
    # Try common encodings, then fallback without crashing
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(fp, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            break

    # last resort: ignore bad bytes
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def is_text_candidate(filename: str) -> bool:
    n = filename.lower().strip()
    # skip obvious binary files
    if n.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf", ".zip", ".rar")):
        return False
    # allow .txt and also files with no extension (like "General-QA")
    ext = os.path.splitext(n)[1]
    return (ext == ".txt") or (ext == "")

def is_sop_file(filename: str) -> bool:
    return "sop" in filename.lower()

def bucket_from_filename(filename: str) -> str:
    n = filename.lower()
    if "both" in n:
        return "both"
    if "mybayut" in n or "bayut" in n:
        return "bayut"
    if "dubizzle" in n:
        return "dubizzle"
    return "general"

def parse_qa_pairs(text: str):
    pairs = []
    pattern = re.compile(
        r"(?im)^\s*Q\s*[:\-–]\s*(.*?)\s*$\n^\s*A\s*[:\-–]\s*(.*?)(?=^\s*Q\s*[:\-–]\s*|\Z)",
        re.DOTALL | re.MULTILINE
    )
    for m in pattern.finditer(text):
        q = re.sub(r"\s+", " ", m.group(1)).strip()
        a = m.group(2).strip()
        a = re.sub(r"\n{3,}", "\n\n", a).strip()
        if q and a:
            pairs.append((q, a))
    return pairs

def normalize_download_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())

def sop_candidates():
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp) and is_sop_file(f):
            out.append(f)
    return sorted(out)

def best_sop_match(user_query: str):
    t = normalize_download_query(user_query)
    if ("download" not in t) and ("sop" not in t) and ("file" not in t):
        return []

    all_sops = sop_candidates()
    if not all_sops:
        return []

    tokens = [x for x in re.split(r"[^a-z0-9]+", t) if x]
    topic_tokens = [x for x in tokens if x not in {"download", "sop", "file", "the", "a", "an"}]

    if ("download" in tokens and "sop" in tokens) and (not topic_tokens):
        return all_sops

    mode = st.session_state.tool_mode

    def pick_by_contains(substrs):
        return [f for f in all_sops if all(s in f.lower() for s in substrs)]

    if any(k in t for k in ["newsletter", "newsletters"]):
        bay = pick_by_contains(["bayut", "newsletters"])
        dub = pick_by_contains(["dubizzle", "newsletters"])
        if mode == "Bayut":
            return bay or []
        if mode == "Dubizzle":
            return dub or []
        return (bay + dub) if (bay or dub) else []

    if any(k in t for k in ["algolia", "location", "locations"]):
        return pick_by_contains(["algolia", "locations"]) or []

    if any(k in t for k in ["pm", "campaign", "campaigns", "performance", "marketing"]):
        bay = pick_by_contains(["bayut", "campaign"])
        dub = pick_by_contains(["dubizzle", "campaign"])
        if mode == "Bayut":
            return bay or []
        if mode == "Dubizzle":
            return dub or []
        return (bay + dub) if (bay or dub) else []

    if any(k in t for k in ["social", "instagram", "posting"]):
        return pick_by_contains(["social", "posting"]) or []

    if any(k in t for k in ["correction", "corrections", "update", "updates", "listing", "listings", "project", "projects"]):
        return [f for f in all_sops if "corrections" in f.lower() or "updates" in f.lower()] or []

    matches = []
    for f in all_sops:
        fn = f.lower()
        if any(tok in fn for tok in topic_tokens):
            matches.append(f)

    return matches

def format_thinking_answer(primary: str, extras: list[str]) -> str:
    out = []
    if primary:
        out.append(primary.strip())
    for ex in extras:
        ex = ex.strip()
        if not ex:
            continue
        if primary and ex.lower() == primary.lower():
            continue
        out.append(ex)
    return "\n\n".join(out[:4]) if out else "No relevant answer found in internal Q&A."

# =====================================================
# EMBEDDINGS
# =====================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# BUILD STORES (Embed question only)
# =====================================================
@st.cache_resource
def build_stores():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("❌ /data folder not found")

    emb = get_embeddings()
    docs_all, docs_bayut, docs_dubizzle = [], [], []

    for fname in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fp) or fname.startswith("."):
            continue

        # ✅ skip SOPs + skip images/binary
        if is_sop_file(fname):
            continue
        if not is_text_candidate(fname):
            continue

        text = read_text(fp)
        if not text.strip():
            continue

        pairs = parse_qa_pairs(text)
        if not pairs:
            continue

        bucket = bucket_from_filename(fname)

        for q, a in pairs:
            doc = Document(page_content=q, metadata={"answer": a, "source": fname, "bucket": bucket})
            docs_all.append(doc)
            if bucket in ("bayut", "both"):
                docs_bayut.append(doc)
            if bucket in ("dubizzle", "both"):
                docs_dubizzle.append(doc)

    if not docs_all:
        raise RuntimeError("❌ No Q&A pairs found. Ensure your files contain 'Q:' and 'A:' blocks.")

    vs_all = FAISS.from_documents(docs_all, emb)
    vs_bayut = FAISS.from_documents(docs_bayut, emb) if docs_bayut else None
    vs_dubizzle = FAISS.from_documents(docs_dubizzle, emb) if docs_dubizzle else None
    return vs_all, vs_bayut, vs_dubizzle

VS_ALL, VS_BAYUT, VS_DUBIZZLE = build_stores()

def pick_store(mode: str):
    if mode == "Bayut":
        return VS_BAYUT
    if mode == "Dubizzle":
        return VS_DUBIZZLE
    return VS_ALL

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 class="center" style="font-weight:900;margin-bottom:6px;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span> AI Content Assistant
    </h1>
    <div class="center" style="color:#666;margin-bottom:14px;">Internal AI Assistant</div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TOOL MODE BUTTONS (CENTERED)
# =====================================================
tool_cols = st.columns([2, 3, 3, 3, 2])
with tool_cols[1]:
    if st.button("General", use_container_width=True, key="btn_tool_general"):
        st.session_state.tool_mode = "General"
with tool_cols[2]:
    if st.button("Bayut", use_container_width=True, key="btn_tool_bayut"):
        st.session_state.tool_mode = "Bayut"
with tool_cols[3]:
    if st.button("Dubizzle", use_container_width=True, key="btn_tool_dubizzle"):
        st.session_state.tool_mode = "Dubizzle"

st.markdown(
    f"<h3 class='center' style='margin-top:18px;margin-bottom:6px;'>{st.session_state.tool_mode} Assistant</h3>",
    unsafe_allow_html=True
)

# =====================================================
# ANSWER MODE BUTTONS (CENTERED) + smaller
# =====================================================
mode_cols = st.columns([5, 2, 2, 5])
with mode_cols[1]:
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("Ultra-Fast", use_container_width=True, key="btn_mode_fast"):
        st.session_state.answer_mode = "Ultra-Fast"
    st.markdown("</div>", unsafe_allow_html=True)

with mode_cols[2]:
    st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
    if st.button("Thinking", use_container_width=True, key="btn_mode_thinking"):
        st.session_state.answer_mode = "Thinking"
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# =====================================================
# INPUT + ASK/CLEAR
# =====================================================
outer = st.columns([1, 6, 1])
with outer[1]:
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")
        bcols = st.columns([1, 1])
        ask = bcols[0].form_submit_button("Ask", use_container_width=True)
        clear = bcols[1].form_submit_button("Clear chat", use_container_width=True)

if clear:
    st.session_state.chat[st.session_state.tool_mode] = []
    st.rerun()

# =====================================================
# ANSWER
# =====================================================
if ask and q:
    sop_files = best_sop_match(q)
    if sop_files:
        st.session_state.chat[st.session_state.tool_mode].append({"type": "download", "q": q, "files": sop_files})
        st.rerun()

    thinking = (st.session_state.answer_mode == "Thinking")
    vs = pick_store(st.session_state.tool_mode)

    if st.session_state.tool_mode == "Bayut" and vs is None:
        st.session_state.chat["Bayut"].append({"type": "qa", "q": q, "a": "No Bayut/MyBayut Q&A files detected."})
        st.rerun()
    if st.session_state.tool_mode == "Dubizzle" and vs is None:
        st.session_state.chat["Dubizzle"].append({"type": "qa", "q": q, "a": "No dubizzle Q&A files detected."})
        st.rerun()

    if thinking:
        with st.spinner("Thinking…"):
            time.sleep(0.35)

    k = 8 if thinking else 4
    results = vs.similarity_search(q, k=k)

    answers = []
    for r in results:
        a = (r.metadata.get("answer") or "").strip()
        if not a:
            continue
        if a.lower() in [x.lower() for x in answers]:
            continue
        answers.append(a)

    if not answers:
        final = "No relevant answer found in internal Q&A."
    else:
        final = answers[0] if not thinking else format_thinking_answer(answers[0], answers[1:])

    st.session_state.chat[st.session_state.tool_mode].append({"type": "qa", "q": q, "a": final})
    st.rerun()

# =====================================================
# CHAT HISTORY (ONLY CURRENT MODE)
# =====================================================
bubble_class = {
    "General": "q-general",
    "Bayut": "q-bayut",
    "Dubizzle": "q-dubizzle",
}[st.session_state.tool_mode]

history = st.session_state.chat.get(st.session_state.tool_mode, [])

for idx, item in enumerate(reversed(history)):
    st.markdown(
        f"<div class='q-bubble {bubble_class}'>{html.escape(item.get('q',''))}</div>",
        unsafe_allow_html=True
    )

    if item.get("type") == "download":
        files = item.get("files", [])
        if not files:
            st.markdown("<div class='answer'>No matching SOP found for your request.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='answer'><b>Download SOP file(s):</b></div>", unsafe_allow_html=True)
            for f in files:
                fp = os.path.join(DATA_DIR, f)
                if not os.path.isfile(fp):
                    continue
                with open(fp, "rb") as bf:
                    st.download_button(
                        label=f"Download: {f}",
                        data=bf,
                        file_name=f,
                        mime="text/plain",
                        key=f"dl_{hashlib.md5((f+str(idx)).encode()).hexdigest()}",
                        use_container_width=False
                    )
    else:
        st.markdown(f"<div class='answer'>{item.get('a','')}</div>", unsafe_allow_html=True)

    st.markdown("---")
