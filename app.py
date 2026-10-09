import streamlit as st
import ollama

st.set_page_config(
    page_title="FinTech Flow Explainer",
    page_icon="💳",
    layout="wide"
)

SYSTEM_INSTRUCTION = """
You are a specialized FinTech Compliance & Transaction Flow Explainer Bot.

Your sole purpose is to explain:
- digital payment workflows
- payment initiation
- transaction verification
- authorization
- settlement processes
- regulatory compliance concepts

Explain concepts clearly and in simple language.

STRICT GUARDRAILS:

1. INFORMATIONAL ONLY:
Provide educational explanations of payment mechanisms and compliance processes.

2. NO FINANCIAL ADVICE:
Do NOT give investment, tax, legal, or personal financial advice.

3. NO PAYMENT PROCESSING:
You cannot execute transactions, process payments, check account balances,
or access personal payment details.

4. BOUNDARY ENFORCEMENT:
If a user asks you to execute a payment, transfer money, check their balance,
or provide financial/investment advice, politely refuse and state:

"I am an informational bot designed only to explain FinTech payment
workflows and compliance processes. I cannot process transactions or
provide financial advice."
"""

MODEL = "qwen2.5:3b"


st.title("💳 FinTech Flow & Compliance Explainer")

st.caption(
    "Learn about digital payment initiation, verification, settlement, "
    "and compliance checks."
)


# Sidebar Quick Test Queries
st.sidebar.header("Sample Queries")

sample_queries = [
    "Explain digital payment process",
    "What are compliance checks?",
    "Explain settlement stages",
    "What is transaction verification?",
]

selected_sample = None

for query in sample_queries:
    if st.sidebar.button(query, use_container_width=True):
        selected_sample = query


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
prompt = (
    st.chat_input("Ask about payment flows or compliance...")
    or selected_sample
)


if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Build messages for Ollama
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    for message in st.session_state.messages:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Generating explanation..."):

            try:

                response = ollama.chat(
                    model=MODEL,
                    messages=messages
                )

                reply = response["message"]["content"]

                st.markdown(reply)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )

            except Exception as e:

                st.error(f"Error connecting to Ollama: {e}")