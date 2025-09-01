import os
import sys
import datetime
import traceback
from flask import Flask, render_template, request, jsonify

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Groq LLM wrapper

# --- Config ---
DB_FAISS_PATH = "vectorstore/db_faiss"
LOG_FILE = "chatbot_log.txt"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

app = Flask(__name__)

# --- Initialize models ---
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
except Exception as e:
    print("❌ Initialization failed:", e)
    db, llm = None, None


def log_interaction(user_query, context_text, prompt, answer, error=None):
    """Log interactions and errors into a text file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"User Query: {user_query}\n")
        if context_text:
            f.write(f"Context Extracted:\n{context_text[:1000]}...\n")
        f.write(f"Final Prompt:\n{prompt}\n")
        if answer:
            f.write(f"Answer:\n{answer}\n")
        if error:
            f.write(f"Error:\n{error}\n")
        f.write("=" * 80 + "\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_query = data.get("query", "").strip()
        custom_template = data.get("template", "").strip()

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # Retrieve docs
        docs = db.similarity_search(user_query, k=3)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt
        if custom_template:
            prompt = custom_template.format(context=context_text, question=user_query)
        else:
            prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

        # Call LLM
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Log interaction
        log_interaction(user_query, context_text, prompt, answer)

        return jsonify({
            "answer": answer,
            "sources": [doc.page_content[:300] for doc in docs]
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        log_interaction(data.get("query", ""), None, None, None, error=error_msg)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
