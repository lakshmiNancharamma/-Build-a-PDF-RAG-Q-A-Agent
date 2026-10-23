from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# PDF file location
PDF_PATH = Path(__file__).parent / "documents" / "archents Leave Policy 2.0.pdf"


# Check whether PDF exists
print("Looking for PDF at:")
print(PDF_PATH)

if not PDF_PATH.exists():
    print("ERROR: PDF file was not found.")
    print("Please make sure the PDF is inside the documents folder.")
    exit()


# 1. Load PDF
loader = PyPDFLoader(str(PDF_PATH))

documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF.")


# 2. Split PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# 3. Create embeddings
# Using a separate embedding model for RAG
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


print("PDF successfully processed and stored in ChromaDB.")