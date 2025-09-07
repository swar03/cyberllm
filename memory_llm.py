import os
import sys

# Suppress TensorFlow logs and oneDNN optimizations if TF is installed
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --- Configuration ---
DATA_FILES = [
    r'C:\Users\Hp\OneDrive\Desktop\cyberllm_with_docs\data\PART 1 - UAE MANUAL.docx',
    r'C:\Users\Hp\OneDrive\Desktop\cyberllm_with_docs\data\Part 2- UAE MANUAL_FV.docx',
    r'C:\Users\Hp\OneDrive\Desktop\cyberllm_with_docs\data\Part 3-Technical Part.docx',
    r'C:\Users\Hp\OneDrive\Desktop\cyberllm_with_docs\data\PART 4- UAE MANUAL_FV.docx',
    r'C:\Users\Hp\OneDrive\Desktop\cyberllm_with_docs\data\Part 5- UAE MANUAL_FV.docx'
]
DB_FAISS_PATH = 'vectorstore/db_faiss'


def load_docx_documents(file_list):
    """Load all DOCX files from a given list of paths"""
    all_documents = []
    for file_path in file_list:
        if file_path.lower().endswith(".docx") and os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            print(f" Loading DOCX: {file_name}")
            loader = Docx2txtLoader(file_path)
            all_documents.extend(loader.load())
        else:
            print(f" Skipping (not found or not DOCX): {file_path}")
    return all_documents


def create_vector_store():
    print("Starting to process DOCX files...")

    # Load all DOCX documents
    documents = load_docx_documents(DATA_FILES)

    if not documents:
        print(" Error: No DOCX documents found.")
        return
    print(f"Successfully loaded {len(documents)} documents/pages.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    print(f" Split documents into {len(text_chunks)} chunks.")

    # Initialize embedding model with error handling
    try:
        print(" Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        print("Error: HuggingFace embeddings require missing dependencies.")
        print(" Run: pip install sentence-transformers torch transformers accelerate")
        sys.exit(1)
    except Exception as e:
        print(f" Unexpected error initializing embeddings: {e}")
        sys.exit(1)

    # Create FAISS vector store
    print(" Creating FAISS vector store... (This may take a moment)")
    db = FAISS.from_documents(text_chunks, embedding_model)

    # Save the vector store locally
    db.save_local(DB_FAISS_PATH)
    print(f" Vector store created successfully and saved at: {DB_FAISS_PATH}")


if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('vectorstore', exist_ok=True)

    # Run vector store creation
    create_vector_store()
