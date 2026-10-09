import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

KNOWLEDGE_FILE = "knowledge/fintech_knowledge.txt"
CHROMA_DIRECTORY = "chroma_db"


# --------------------------------------------------
# 2. Load knowledge file
# --------------------------------------------------

print("Loading knowledge file...")

loader = TextLoader(
    KNOWLEDGE_FILE,
    encoding="utf-8"
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")


# --------------------------------------------------
# 3. Split document into smaller chunks
# --------------------------------------------------

print("Splitting knowledge into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 4. Create embedding model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 5. Create Chroma vector database
# --------------------------------------------------

print("Creating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIRECTORY
)

print("Knowledge base created successfully!")
print(f"Stored in: {os.path.abspath(CHROMA_DIRECTORY)}")