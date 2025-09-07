import os
import datetime
import traceback
import streamlit as st
import torch
from dotenv import load_dotenv  # <-- 1. IMPORT THE LIBRARY
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Load Environment Variables ---
load_dotenv()  # <-- 2. LOAD THE .ENV FILE

# --- Config ---
DB_FAISS_PATH = "vectorstore/db_faiss"
LOG_FILE = "chatbot_log.txt"

# --- Page Configuration ---
st.set_page_config(page_title="🔍 Cybercrime Investigation Assistant ", page_icon="🤖", layout="wide")

# --- Model and Resource Loading ---
torch.set_default_device('cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def initialize_models():
    """Initialize FAISS database and ChatGroq LLM with error handling."""
    try:
        # This line now works for BOTH local and deployment
        groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file (for local) or Streamlit secrets (for deployment).")
            return None, None
            
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
            st.info("👉 Please ensure the 'vectorstore' directory exists.")
            return None, None
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        return db, llm
        
    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
        st.code(traceback.format_exc())
        return None, None

# (The rest of your code remains exactly the same)

def log_interaction(user_query, context_text, prompt, answer, error=None):
    """Log interactions and errors into a text file."""
    try:
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
    except Exception as e:
        st.warning(f"Failed to log interaction: {e}")

# --- Streamlit UI ---
st.title("🔍 Cybercrime Investigation Assistant")
st.markdown("Ask questions about cybercrime investigation procedures, UAE laws, and digital evidence handling.")

db, llm = initialize_models()

if db is None or llm is None:
    st.error("❌ Application failed to start. Please check the configuration and logs.")
    st.stop()

st.success("✅ Models initialized successfully!")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.info("""
    **Model Info:**
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **LLM:** `llama-3.1-70b-versatile` (GROQ)
    - **Vector Store:** FAISS
    """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤔 Thinking..."):
            try:
                docs = db.similarity_search(user_query, k=3)
                context_text = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                
                log_interaction(user_query, context_text, prompt, answer)
                
                full_response = answer
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content)
                        st.markdown("---")
                
                message_placeholder.markdown(full_response)

            except Exception as e:
                error_msg = traceback.format_exc()
                log_interaction(user_query, None, None, None, error=error_msg)
                full_response = f"❌ An error occurred: {e}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Powered by GROQ, LangChain & FAISS</div>", unsafe_allow_html=True)