import streamlit as st
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinTech Flow",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen2.5:3b"
CHROMA_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MAX_HISTORY_MESSAGES = 6
MAX_NEW_TOKENS = 500


# ============================================================
# HTML RENDER HELPER
# ============================================================

def clean_html(html: str) -> str:
    """
    Strip leading/trailing whitespace from every line and drop blank
    lines. Markdown treats 4+ space indents and blank lines as code
    blocks / block breaks, which is what made raw HTML show up as text.
    """
    lines = [line.strip() for line in html.splitlines()]
    return "\n".join(line for line in lines if line)


def render_html(html: str, sidebar: bool = False):
    target = st.sidebar if sidebar else st
    target.markdown(clean_html(html), unsafe_allow_html=True)


# ============================================================
# GLOBAL CSS
# ============================================================

render_html("""
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 10%, rgba(100, 70, 210, 0.18), transparent 30%),
        radial-gradient(circle at 10% 85%, rgba(0, 150, 255, 0.10), transparent 30%),
        #070914;
}
.main { background: transparent; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0e1c 0%, #080a14 100%);
    border-right: 1px solid rgba(130, 120, 255, 0.12);
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 285px;
    padding: 42px;
    margin-bottom: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(22, 26, 58, 0.97), rgba(10, 13, 29, 0.97));
    border: 1px solid rgba(125, 110, 255, 0.20);
    box-shadow: 0 25px 70px rgba(0, 0, 0, 0.35), inset 0 0 60px rgba(90, 70, 200, 0.05);
}
.hero-title {
    position: relative;
    z-index: 5;
    margin: 0;
    font-size: 48px;
    line-height: 1.1;
    font-weight: 850;
    background: linear-gradient(90deg, #ffffff, #b4c0ff, #d59cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    position: relative;
    z-index: 5;
    margin-top: 8px;
    color: #eef1ff;
    font-size: 29px;
    font-weight: 700;
}
.hero-description {
    position: relative;
    z-index: 5;
    max-width: 700px;
    margin-top: 13px;
    color: #aeb5cc;
    font-size: 16px;
    line-height: 1.6;
}

/* FLOATING COINS */
.coin {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: radial-gradient(circle at 32% 25%, #fff0ad 0%, #ffd452 25%, #e6a51b 55%, #a96408 82%, #623703 100%);
    border: 3px solid rgba(255, 236, 169, 0.75);
    color: #704000;
    font-family: Arial, sans-serif;
    font-weight: 900;
    box-shadow: inset -8px -9px 12px rgba(70, 35, 0, 0.35), inset 6px 5px 9px rgba(255, 255, 255, 0.45), 0 12px 40px rgba(255, 170, 20, 0.25);
    z-index: 2;
    pointer-events: none;
    animation: coinFloat 5s ease-in-out infinite;
}
.coin1 { width: 72px; height: 72px; right: 10%; top: 32px; font-size: 31px; animation-delay: 0s; }
.coin2 { width: 52px; height: 52px; right: 3%; bottom: 28px; font-size: 23px; animation-delay: 1.2s; }
.coin3 { width: 44px; height: 44px; right: 26%; bottom: 17px; font-size: 19px; animation-delay: 2s; }
.coin4 { width: 38px; height: 38px; right: 34%; top: 35px; font-size: 16px; animation-delay: 2.8s; }

@keyframes coinFloat {
    0% { transform: translateY(0px) rotate(-8deg); }
    50% { transform: translateY(-14px) rotate(8deg); }
    100% { transform: translateY(0px) rotate(-8deg); }
}

/* BADGES */
.badges {
    position: relative;
    z-index: 5;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}
.badge {
    padding: 8px 14px;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(150, 150, 255, 0.14);
    color: #cbd2ee;
    font-size: 13px;
}

/* SIDEBAR BRAND */
.sidebar-brand { padding: 8px 4px 20px 4px; }
.sidebar-brand-title { color: #f5f7ff; font-size: 24px; font-weight: 800; }
.sidebar-brand-subtitle { color: #8992ad; font-size: 13px; margin-top: 2px; }

/* RAG STATUS */
.rag-status {
    margin-top: 22px;
    padding: 15px;
    border-radius: 17px;
    background: linear-gradient(135deg, rgba(12, 70, 55, 0.30), rgba(10, 35, 35, 0.25));
    border: 1px solid rgba(55, 210, 150, 0.20);
}
.rag-title { color: #e9fff6; font-size: 14px; font-weight: 700; }
.rag-description { margin-top: 5px; color: #91a19e; font-size: 12px; line-height: 1.5; }
.rag-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    margin-right: 7px;
    border-radius: 50%;
    background: #39e59a;
    box-shadow: 0 0 12px rgba(57, 229, 154, 0.85);
}

/* TRANSACTION FLOW */
.section-title { margin-top: 35px; margin-bottom: 6px; color: #f5f7ff; font-size: 25px; font-weight: 750; }
.section-subtitle { margin-bottom: 18px; color: #8992ad; font-size: 14px; }
.flow-container {
    display: flex;
    align-items: stretch;
    gap: 8px;
    width: 100%;
    padding: 10px 2px 25px 2px;
    overflow-x: auto;
}
.flow-step {
    min-width: 125px;
    flex: 1;
    padding: 18px 12px;
    text-align: center;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(30, 35, 67, 0.96), rgba(13, 17, 34, 0.96));
    border: 1px solid rgba(120, 120, 255, 0.16);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
}
.flow-number { margin-bottom: 6px; color: #7e8cff; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
.flow-icon { margin-bottom: 8px; font-size: 27px; }
.flow-title { color: #f5f7ff; font-size: 13px; font-weight: 700; }
.flow-arrow { display: flex; align-items: center; color: #8178ed; font-size: 21px; }

/* FOOTER */
.app-footer { margin-top: 32px; padding: 20px; text-align: center; color: #626b84; font-size: 12px; }

/* STREAMLIT CLEANUP */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }
</style>
""")


# ============================================================
# SYSTEM GUARDRAILS
# ============================================================

SYSTEM_INSTRUCTION = """
You are a specialized FinTech Compliance & Transaction Flow Explainer Bot.

Your purpose is to explain:
- Digital payment workflows
- UPI
- Payment initiation
- Transaction verification
- Authentication
- Authorization
- Settlement
- Reconciliation
- KYC
- AML
- CFT
- Digital payment security
- Fraud prevention
- Payment disputes

IMPORTANT RAG RULE:
Answer questions using the provided KNOWLEDGE CONTEXT whenever possible.
Do NOT invent regulatory requirements, transaction limits, bank policies, or payment-system rules.
If the provided knowledge context does not contain enough information to answer the question,
clearly say that the available knowledge base does not contain enough information.
Do not pretend to have access to real banking systems.

STRICT GUARDRAILS:

1. INFORMATIONAL ONLY:
Provide educational explanations of payment mechanisms and compliance processes.

2. NO FINANCIAL ADVICE:
Do NOT provide investment, tax, legal, or personalized financial advice.

3. NO PAYMENT PROCESSING:
You cannot execute transactions, transfer money, check account balances, or access personal payment details.

4. NO SENSITIVE CREDENTIALS:
Never ask the user for: UPI PIN, OTP, ATM PIN, Password, CVV, or Banking credentials.

5. NO REGULATORY BYPASS:
Do not provide instructions for bypassing KYC, AML, CFT, sanctions, fraud controls, or other regulatory requirements.

6. TRANSACTION REQUESTS:
If a user asks you to execute a payment, transfer money, check their balance, or access their bank account,
explain that you are informational only and cannot perform the action.
"""


# ============================================================
# LOAD EMBEDDINGS + CHROMA
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Opening knowledge base...")
def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=load_embeddings(),
    )


def get_chunk_count():
    try:
        return load_vectorstore()._collection.count()
    except Exception:
        return None


# ============================================================
# RETRIEVE KNOWLEDGE
# ============================================================

def retrieve_context(question: str) -> str:
    documents = load_vectorstore().similarity_search(question, k=TOP_K)
    return "\n\n".join(doc.page_content for doc in documents)


# ============================================================
# GENERATE ANSWER (STREAMING)
# ============================================================

def build_messages(question, context, history):
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    for message in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": message["role"], "content": message["content"]})

    messages.append({
        "role": "user",
        "content": f"""KNOWLEDGE CONTEXT:

{context}

--------------------------------------------------

USER QUESTION:

{question}

--------------------------------------------------

Instructions:
Answer the user's question using the knowledge context above.
Explain the answer clearly and simply.
Keep the answer concise and focused.
If the context does not contain enough information, say so instead of inventing information.""",
    })
    return messages


def stream_answer(question, context, history):
    stream = ollama.chat(
        model=MODEL,
        messages=build_messages(question, context, history),
        options={"num_predict": MAX_NEW_TOKENS},
        stream=True,
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield piece


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">
<div class="coin coin1">₹</div>
<div class="coin coin2">₹</div>
<div class="coin coin3">₹</div>
<div class="coin coin4">₹</div>
<div class="hero-title"> FinTech Flow</div>
<div class="hero-subtitle">Compliance & Transaction Explainer</div>
<div class="hero-description">
Understand digital payments, UPI, verification, settlement, KYC, AML and compliance
through an AI-powered local knowledge system.
</div>
<div class="badges">
<div class="badge">💳 Digital Payments</div>
<div class="badge">🛡️ Compliance</div>
<div class="badge">🔐 Security</div>
<div class="badge">🧠 RAG Powered</div>
<div class="badge">🤖 Local AI</div>
</div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

render_html("""
<div class="sidebar-brand">
<div class="sidebar-brand-title">💳 FinTech Flow</div>
<div class="sidebar-brand-subtitle">Compliance Explainer</div>
</div>
""", sidebar=True)

st.sidebar.markdown("### ⚡ Popular Topics")

sample_queries = [
    "What is the role of NPCI in UPI?",
    "Explain the UPI payment process",
    "What is transaction authentication?",
    "What is settlement?",
    "What is KYC?",
    "What is AML?",
    "Difference between fraud and compliance?",
]

selected_sample = None
for query in sample_queries:
    if st.sidebar.button(query, use_container_width=True):
        selected_sample = query

chunk_count = get_chunk_count()
chunk_text = f"{chunk_count} knowledge chunks" if chunk_count is not None else "Knowledge base not found"

render_html(f"""
<div class="rag-status">
<div class="rag-title"><span class="rag-dot"></span>RAG Enabled</div>
<div class="rag-description">Local Chroma Knowledge Base<br>{chunk_text}</div>
</div>
""", sidebar=True)

st.sidebar.caption("🤖 Qwen 2.5 3B • Ollama")


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

prompt = st.chat_input("Ask about UPI, payment flows, KYC, AML or compliance...") or selected_sample


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    with st.spinner("Searching FinTech knowledge base..."):
        try:
            context = retrieve_context(prompt)
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            st.stop()

    if not context.strip():
        context = "(No relevant knowledge found.)"

    # Generate (streamed)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_answer(prompt, context, st.session_state.messages)
            )
        except Exception as e:
            st.error(
                f"Error connecting to Ollama: {e}\n\n"
                f"Make sure Ollama is running and the model is pulled: `ollama pull {MODEL}`"
            )
            st.stop()

    # Save only after success
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})


# ============================================================
# TRANSACTION FLOW
# ============================================================

render_html("""
<div class="section-title">🔄 Digital Payment Transaction Flow</div>
<div class="section-subtitle">Explore the major stages involved in a typical digital payment.</div>
""")

flow_steps = [
    ("1", "🚀", "Initiation"),
    ("2", "🔎", "Verification"),
    ("3", "🔐", "Authentication"),
    ("4", "🛡️", "Compliance"),
    ("5", "⚙️", "Processing"),
    ("6", "🏦", "Settlement"),
    ("7", "✅", "Confirmation"),
]

parts = ['<div class="flow-container">']
for index, icon, title in flow_steps:
    parts.append(
        f'<div class="flow-step">'
        f'<div class="flow-number">STEP {index}</div>'
        f'<div class="flow-icon">{icon}</div>'
        f'<div class="flow-title">{title}</div>'
        f'</div>'
    )
    if index != flow_steps[-1][0]:
        parts.append('<div class="flow-arrow">→</div>')
parts.append("</div>")

render_html("\n".join(parts))


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="app-footer">
🔐 Educational FinTech Assistant &nbsp; • &nbsp; Local AI + RAG &nbsp; • &nbsp; No transaction processing
</div>
""")