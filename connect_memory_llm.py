import os
import sys
import datetime
import traceback
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Groq LLM wrapper
load_dotenv()
# --- Config ---
DB_FAISS_PATH = "vectorstore/db_faiss"
LOG_FILE = "chatbot_log.txt"

# Set your Groq API key here (or via environment variable)
groq_api_key = os.getenv("GROQ_API_KEY")

def log_interaction(user_query, context_text, prompt, answer, error=None):
    """Log interactions and errors into a text file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"User Query: {user_query}\n")
        if context_text:
            f.write(f"Context Extracted:\n{context_text[:1000]}...\n")
        f.write(f"Final Prompt:\n{prompt}\n")
        if answer:
            f.write(f"Answer:\n{answer}\n")
        if error:
            f.write(f"Error:\n{error}\n")
        f.write("="*80 + "\n")


def main():
    if not os.path.exists(DB_FAISS_PATH):
        print(f"❌ Error: Vector store not found at {DB_FAISS_PATH}")
        print("👉 Please run 'create_memory_llm.py' first.")
        sys.exit(1)

    # Initialize Groq LLM
    try:
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",  # or "llama-3.2-90b-text"
            api_key=groq_api_key
        )
    except Exception as e:
        print("❌ Failed to initialize Groq LLM.")
        traceback.print_exc()
        sys.exit(1)

    # Load embeddings + FAISS
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ Failed to load FAISS vector store.")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Cyber Crime Investigation Chatbot (Groq API) ---")
    print("Ask a question to begin. Type 'exit' to quit.")
    print("Type '/prompt' to enter a custom prompt template.")

    custom_template = None  # Default is None → uses built-in template

    while True:
        try:
            user_query = input("\nYour Query: ").strip()
            if user_query.lower() == "exit":
                break

            # Allow user to define a custom prompt template
            if user_query.lower() == "/prompt":
                print("\n👉 Enter your custom prompt template.")
                print("   Use {context} for retrieved text and {question} for user query.")
                custom_template = input("Custom Template: ").strip()
                print("✅ Custom template saved!")
                continue

            # Retrieve relevant docs
            docs = db.similarity_search(user_query, k=3)
            context_text = "\n\n".join(doc.page_content for doc in docs)

            # Build final prompt
            if custom_template:
                prompt = custom_template.format(context=context_text, question=user_query)
            else:
                prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

            # Call LLM
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            # Show result
            print("\nAnswer:", answer)
            print("\n--- Sources Used ---")
            for i, doc in enumerate(docs):
                print(f"Source {i+1}:\n{doc.page_content[:350]}...\n")

            # Log interaction
            log_interaction(user_query, context_text, prompt, answer)

        except Exception as e:
            error_msg = traceback.format_exc()
            print("❌ An error occurred while processing your query.")
            print(error_msg)
            log_interaction(user_query, None, None, None, error=error_msg)


if __name__ == "__main__":
    main()
