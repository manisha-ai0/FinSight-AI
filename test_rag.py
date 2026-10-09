from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_DIRECTORY = "chroma_db"


# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load existing Chroma database
vectorstore = Chroma(
    persist_directory=CHROMA_DIRECTORY,
    embedding_function=embeddings
)


# Test question
question = "What is the role of NPCI in UPI?"


# Retrieve the 3 most relevant chunks
results = vectorstore.similarity_search(
    question,
    k=3
)


print("\n===== RETRIEVED INFORMATION =====\n")

for i, document in enumerate(results, start=1):

    print(f"--- Result {i} ---")
    print(document.page_content)
    print()