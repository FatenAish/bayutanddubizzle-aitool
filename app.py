import os
import streamlit as st

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Internal AI Assistant", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===============================
# SESSION STATE
# ===============================
if "chat" not in st.session_state:
    st.session_state.chat = {
        "Bayut": [],
        "dubizzle": [],
        "General": []
    }

# ===============================
# STYLE (ChatGPT bubbles)
# ===============================
st.markdown("""
<style>
.chat-container { display: flex; flex-direction: column; gap: 12px; }
.user-bubble {
    align-self: flex-end;
    background: #DCF8C6;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
}
.bot-bubble {
    align-self: flex-start;
    background: #F1F0F0;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TOOL DETECTION
# ===============================
def detect_tool(q: str) -> str:
    q = q.lower()
    if "dubizzle" in q:
        return "dubizzle"
    if "bayut" in q:
        return "Bayut"
    return "General"

# ===============================
# LOAD QA FILES
# ===============================
def load_qa_files(tool: str):
    qa_pairs = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith("-qa.txt"):
            continue

        name = f.lower()
        if tool == "Bayut" and "bayut" not in name:
            continue
        if tool == "dubizzle" and "dubizzle" not in name:
            continue

        with open(os.path.join(DATA_DIR, f), encoding="utf-8") as file:
            lines = file.read().splitlines()

        q, a = None, []
        for line in lines:
            if line.startswith("Q:"):
                if q and a:
                    qa_pairs.append((q, " ".join(a)))
                q = line.replace("Q:", "").strip()
                a = []
            elif line.startswith("A:"):
                a.append(line.replace("A:", "").strip())
            elif q:
                a.append(line.strip())

        if q and a:
            qa_pairs.append((q, " ".join(a)))

    return qa_pairs

# ===============================
# SMART ANSWER (QA ONLY)
# ===============================
def answer_question(question, tool):
    qa_pairs = load_qa_files(tool)
    question_l = question.lower()

    # Exact / strong match first
    for q, a in qa_pairs:
        if question_l in q.lower():
            return a

    # Keyword overlap fallback
    q_words = set(question_l.split())
    scored = []
    for q, a in qa_pairs:
        score = len(q_words & set(q.lower().split()))
        if score > 0:
            scored.append((score, a))

    if scored:
        return sorted(scored, reverse=True)[0][1]

    return "I don’t have a clear answer for this in the available Q&A files."

# ===============================
# SIDEBAR — DOWNLOAD SOPs
# ===============================
with st.sidebar:
    st.markdown("### 📥 Download SOPs")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith(".txt") and "qa" not in f.lower():
            with open(os.path.join(DATA_DIR, f), "rb") as file:
                st.download_button(
                    label=f,
                    data=file,
                    file_name=f,
                    mime="text/plain"
                )

# ===============================
# TITLE
# ===============================
st.markdown("""
<h1 style="text-align:center;font-weight:800;">Internal AI Assistant</h1>
<p style="text-align:center;color:#666;">Fast internal assistant</p>
""", unsafe_allow_html=True)

# ===============================
# INPUT
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    c1, c2, c3 = st.columns([2, 1, 1])
    ask = c2.form_submit_button("Ask")
    clear = c3.form_submit_button("Clear chat")

# ===============================
# ACTIONS
# ===============================
if ask and q.strip():
    tool = detect_tool(q)
    answer = answer_question(q, tool)
    st.session_state.chat[tool].append({"q": q, "a": answer})
    st.rerun()

if clear:
    tool = detect_tool(q) if q else "General"
    st.session_state.chat[tool] = []
    st.rerun()

# ===============================
# CHAT DISPLAY
# ===============================
tool = detect_tool(q) if q else "General"

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for item in st.session_state.chat[tool]:
    st.markdown(f"""
    <div class="user-bubble">{item['q']}</div>
    <div class="bot-bubble">{item['a']}</div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
