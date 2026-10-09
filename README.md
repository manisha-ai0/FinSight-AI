
# FinSight AI — Financial Transaction Insights & Compliance Explainer
**Financial Clarity. Intelligent Insights.**

An AI-powered FinTech assistant that helps users understand digital payment workflows, UPI transactions, transaction verification, settlement processes, and financial compliance concepts through Retrieval-Augmented Generation (RAG) and a locally running language model.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black?logo=ollama)
![Chroma](https://img.shields.io/badge/Vector_DB-ChromaDB-purple)

## 📌 Overview

Digital payment and compliance workflows can be difficult to understand. FinTech Flow makes these concepts easier to learn through conversational explanations grounded in a local knowledge base.

The application retrieves relevant information from its knowledge base and provides context-aware answers using a locally running language model.

## 🎯 Problem Statement

Understanding digital payment systems involves multiple stages, including payment initiation, verification, authentication, processing, settlement, and compliance checks.

This project aims to simplify these concepts through an interactive AI assistant that explains how payment workflows operate.

## ✨ Key Features

- 🤖 **Local AI chatbot:** Uses Qwen 2.5 3B through Ollama.
- 🧠 **Retrieval-Augmented Generation (RAG):** Retrieves relevant knowledge before generating answers.
- 🔎 **Semantic search:** Finds relevant information using sentence-transformer embeddings.
- 🗄️ **Vector database:** Stores and searches knowledge chunks using Chroma.
- 💬 **Interactive interface:** Provides a conversational UI built with Streamlit.
- 📚 **FinTech knowledge base:** Covers UPI, NPCI, KYC, AML, settlement, authentication, and fraud prevention.
- 🛡️ **Safety boundaries:** Designed for educational explanations, not payment execution.
- 💻 **Local inference:** Runs the language model locally through Ollama.

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Frontend | Streamlit |
| Language model | Qwen 2.5 3B |
| Model runtime | Ollama |
| RAG framework | LangChain |
| Vector database | Chroma |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Knowledge format | Text |
| Environment management | Conda |

## 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │         User         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Streamlit UI       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Python Application │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Chroma Vector DB    │
                  │  Semantic Retrieval  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Retrieved Context    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Qwen 2.5 3B / Ollama │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Generated Explanation│
                  └──────────────────────┘
```

### How It Works

1. The user enters a question in the Streamlit interface.
2. The application converts the question into a semantic search query.
3. Chroma retrieves relevant chunks from the FinTech knowledge base.
4. The retrieved context is provided to the language model.
5. Qwen generates an explanation using the supplied context and system instructions.
6. The answer is displayed in the chat interface.

## 📚 Knowledge Domains

The knowledge base is designed to cover:

- Unified Payments Interface (UPI)
- National Payments Corporation of India (NPCI)
- Payment initiation and verification
- Transaction authentication and authorization
- Payment processing
- Settlement and reconciliation
- Know Your Customer (KYC)
- Anti-Money Laundering (AML)
- Countering the Financing of Terrorism (CFT)
- Digital payment security
- Fraud prevention
- Payment disputes

## 📂 Project Structure

```text
FinTech-Flow/
│
├── app.py
├── rag_app.py
├── build_knowledge.py
├── test.py
├── test_rag.py
├── .gitignore
├── README.md
│
└── Knowledge/
    └── fintech_knowledge.txt
```

The knowledge folder name and file path should match the actual names in the repository.

The generated `chroma_db/` directory is created locally when the knowledge base is built and is excluded from version control.

## ⚙️ Prerequisites

Install the following before running the project:

- Python 3.12
- Conda
- Git
- Ollama

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/manisha-ai0/FinTech-Flow.git
cd FinTech-Flow
```

### 2. Create and activate the Conda environment

```bash
conda create -n langchain_env python=3.12 -y
conda activate langchain_env
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install streamlit ollama langchain langchain-community langchain-chroma langchain-huggingface langchain-text-splitters chromadb sentence-transformers
```

### 4. Install and prepare Ollama

Install Ollama from:

https://ollama.com/

Download the model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before starting the application.

### 5. Build the knowledge base

Run the following command from the project root:

```bash
python build_knowledge.py
```

This loads the knowledge document, splits it into chunks, generates embeddings, and stores them in Chroma.

Build the knowledge base again only when needed, such as after changing the source knowledge file or rebuilding the vector database.

### 6. Run the chatbot

```bash
streamlit run rag_app.py
```

Open the local URL displayed in the terminal. The default is:

http://localhost:8501

## 🧪 Testing

The repository includes scripts for testing the application and knowledge retrieval:

```bash
python test.py
```

```bash
python test_rag.py
```

These scripts are intended to help verify model connectivity and knowledge retrieval. Their exact coverage depends on the current implementation.

## 🔐 Safety and Limitations

FinTech Flow is an educational prototype.

- It does not execute payments or money transfers.
- It does not access bank accounts or account balances.
- It should never request UPI PINs, OTPs, CVVs, passwords, or banking credentials.
- It is not a substitute for official banking documentation or regulatory guidance.
- Generated answers can be incomplete or inaccurate and should be independently verified.
- It is not a production banking, fraud-detection, or regulatory-compliance system.

## 🔮 Future Improvements

- Add citations linking answers to retrieved knowledge sources.
- Expand and validate the FinTech knowledge base.
- Evaluate retrieval quality and answer accuracy.
- Add automated tests for guardrails and unsupported requests.
- Improve the transaction-flow visualization.
- Add deployment instructions and a hosted demo.
- Improve performance and user experience.

## 👩‍💻 Author

**Manisha Rani**

GitHub: [@manisha-ai0](https://github.com/manisha-ai0)


## 📄 License

This project is licensed under the [MIT License](LICENSE).

See the [LICENSE](LICENSE) file for the full license terms.


---

*Developed as a FinTech and Generative AI learning project.*
