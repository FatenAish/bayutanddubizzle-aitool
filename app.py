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
    st.session_state.chat = []

# ===============================
# HELPERS
# ===============================
def is_download_request(q: str) -> bool:
    q = q.lower()
    triggers = ["download", "sop", "file", "send", "give me"]
    return any(t in q for t in triggers)

def find_matching_sop(q: str):
    q = q.lower()
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".txt") and "qa" not in f.lower():
            if any(word in f.lower() for word in q.split()):
                return f
    return None

def load_qa_pairs():
    pairs = []
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith("-qa.txt"):
            continue
        with open(os.path.join(DATA_DIR, f), encoding="utf-8") as file:
            lines = file.read().splitlines()

        q, a = None, []
        for line in lines:
            if line.startswith("Q:"):
                if q and a:
                    pairs.append((q, " ".join(a)))
                q = line.replace("Q:", "").strip()
                a = []
            elif line.startswith("A:"):
                a.append(line.replace("A:", "").strip())
            elif q:
                a.append(line.strip())

        if q and a:
            pairs.append((q, " ".join(a)))

    return pairs

def answer_from_qa(question: str):
    question_l = question.lower()
    qa = load_qa_pairs()

    for q, a in qa:
        if question_l in q.lower():
            return a

    words = set(question_l.split())
    scored = []
    for q, a in qa:
        score = len(words & set(q.lower().split()))
        if score > 0:
            scored.append((score, a))

    if scored:
        return sorted(scored, reverse=True)[0][1]

    return "I don’t have a clear answer for this in the available Q&A."

# ===============================
# TITLE (UNCHANGED)
# ===============================
st.markdown("""
<h1 style="text-align:center;font-weight:800;">Internal AI Assistant</h1>
<p style="text-align:center;color:#666;">Fast internal assistant</p>
""", unsafe_allow_html=True)

# ===============================
# INPUT (UNCHANGED LAYOUT)
# ===============================
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask a question")

    col1, col2, col3 = st.columns([2, 1, 1])
    ask = col2.form_submit_button("Ask")
    clear = col3.form_submit_button("Clear chat")

# ===============================
# ACTIONS
# ===============================
if ask and q.strip():
    if is_download_request(q):
        sop = find_matching_sop(q)
        if sop:
            st.session_state.chat.append({
                "q": q,
                "a": f"📥 You can download the SOP below:",
                "file": sop
            })
        else:
            st.session_state.chat.append({
                "q": q,
                "a": "I couldn’t find a matching SOP to download."
            })
    else:
        answer = answer_from_qa(q)
        st.session_state.chat.append({"q": q, "a": answer})

    st.rerun()

if clear:
    st.session_state.chat = []
    st.rerun()

# ===============================
# CHAT RENDER (DESIGN PRESERVED)
# ===============================
for item in st.session_state.chat:
    st.markdown(f"""
    <div style="background:#DCF8C6;padding:12px;border-radius:12px;margin-bottom:6px;">
    {item['q']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F1F0F0;padding:12px;border-radius:12px;margin-bottom:12px;">
    {item['a']}
    </div>
    """, unsafe_allow_html=True)

    if "file" in item:
        with open(os.path.join(DATA_DIR, item["file"]), "rb") as f:
            st.download_button(
                label=f"Download {item['file']}",
                data=f,
                file_name=item["file"],
                mime="text/plain"
            )
