import os
import re
import html
import time
import streamlit as st

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bayut & Dubizzle Internal Assistant",
    layout="wide"
)

# ===============================
# UI CSS
# ===============================
st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { gap: 0.12rem; }
      button { white-space: nowrap !important; }

      .mode-title{
        font-size: 20px !important;
        font-weight: 750 !important;
        margin: 6px 0;
      }

      .question-wrap{
        max-width: 980px;
        margin: 10px auto;
      }

      .question-wrap [data-testid="stForm"]{
        border: 1px solid #E7E9EE;
        border-radius: 12px;
        padding: 16px 18px 10px 18px;
        background: #fff;
      }

      .question-wrap div[data-testid="stTextInput"] > label { display:none; }
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
if "chat" not in st.session_state:
    st.session_state.chat = {}

st.session_state.chat.setdefault("General", [])
st.session_state.chat.setdefault("Bayut", [])
st.session_state.chat.setdefault("Dubizzle", [])

# ===============================
# EMBEDDINGS (LOCAL ONLY)
# ===============================
class LocalEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

@st.cache_resource
def get_embeddings():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/models"
    )
    return LocalEmbeddings(model)

# ===============================
# HELPERS
# ===============================
def read_text(fp):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def keyword_overlap(a, b):
    return len(set(re.findall(r"\w+", a.lower())) & set(re.findall(r"\w+", b.lower())))

def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"turn\d+file\d+", "", text)
    text = re.sub(r"filecite", "", text, flags=re.I)
    text = re.sub(r"\bQ:\s*|\bA:\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def bubble_style(mode):
    if mode == "Bayut":
        return "background:#EAF7F1;border:1px solid #BFE6D5;"
    if mode == "Dubizzle":
        return "background:#FCEBEC;border:1px solid #F3C1C5;"
    return "background:#F5F6F8;border:1px solid #E2E5EA;"

# ===============================
# Q/A PARSER
# ===============================
QA_RE = re.compile(r"Q\s*[:\-]\s*(.*?)\nA\s*[:\-]\s*(.*?)(?=\nQ|\Z)", re.S | re.I)

def parse_qa(text):
    return [(q.strip(), a.strip()) for q, a in QA_RE.findall(text)]

def list_txt_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

# ===============================
# INDEX LOADERS
# ===============================
@st.cache_resource
def load_qa_index(mode):
    docs = []
    for fp in list_txt_files():
        raw = read_text(fp)
        for q, a in parse_qa(raw):
            docs.append(Document(page_content=q, metadata={"answer": a}))

    if not docs:
        return None

    return FAISS.from_documents(docs, get_embeddings())

@st.cache_resource
def load_sop_index():
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    docs = []
    for fp in list_txt_files():
        raw = read_text(fp)
        for c in splitter.split_text(raw):
            docs.append(Document(page_content=c))
    return FAISS.from_documents(docs, get_embeddings()) if docs else None

# ===============================
# ANSWERING
# ===============================
def smart_answer(q, qa_index, sop_index):
    if not q:
        return ""

    if qa_index:
        res = qa_index.similarity_search(q, k=5)
        if res:
            return clean_answer(res[0].metadata.get("answer", ""))

    if sop_index:
        res = sop_index.similarity_search(q, k=3)
        if res:
            return clean_answer(res[0].page_content)

    return "I couldn’t find a clear answer to that."

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    tool_mode = st.radio("Mode", ["General", "Bayut", "Dubizzle"])
    answer_mode = st.radio("Answer mode", ["Ultra-Fast", "Thinking"])

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;font-weight:800;">
      <span style="color:#0E8A6D;">Bayut</span> &
      <span style="color:#D71920;">Dubizzle</span>
      AI Content Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

# ===============================
# QUESTION UI
# ===============================
st.markdown('<div class="question-wrap">', unsafe_allow_html=True)

if tool_mode == "Bayut":
    st.markdown('<div class="mode-title">Ask <span style="color:#0E8A6D;">Bayut</span> Anything</div>', unsafe_allow_html=True)
elif tool_mode == "Dubizzle":
    st.markdown('<div class="mode-title">Ask <span style="color:#D71920;">Dubizzle</span> Anything</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mode-title">General Assistant</div>', unsafe_allow_html=True)

with st.form("ask", clear_on_submit=True):
    q = st.text_input("", placeholder="Type your question here…")
    c1, c2 = st.columns([1, 1])
    ask = c1.form_submit_button("Ask")
    clear = c2.form_submit_button("Clear chat")

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# CLEAR
# ===============================
if clear:
    st.session_state.chat[tool_mode] = []
    st.rerun()

# ===============================
# ASK
# ===============================
if ask and q:
    if answer_mode == "Thinking":
        with st.spinner("Thinking…"):
            time.sleep(1.2)

    qa_index = load_qa_index(tool_mode)
    sop_index = load_sop_index()
    answer = smart_answer(q, qa_index, sop_index)

    st.session_state.chat[tool_mode].append({"q": q, "a": answer})
    st.rerun()

# ===============================
# CHAT (NEWEST FIRST)
# ===============================
for item in reversed(st.session_state.chat[tool_mode]):
    st.markdown(
        f"<div style='{bubble_style(tool_mode)} padding:12px;border-radius:10px;'><b>Q:</b> {html.escape(item['q'])}</div>",
        unsafe_allow_html=True
    )
    st.markdown(clean_answer(item["a"]))
    st.markdown("---")
