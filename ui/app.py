import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import build_rag_pipeline


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production-Grade RAG Chatbot")


# ------------------------------
# LOAD RAG PIPELINE (CACHE)
# ------------------------------
@st.cache_resource
def load_pipeline():
    return build_rag_pipeline()

rag = load_pipeline()


# ------------------------------
# SESSION STATE (CHAT HISTORY)
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------
# USER INPUT
# ------------------------------
query = st.chat_input("Ask something...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Run RAG pipeline
            result = rag.run(query)

            answer = result["answer"]
            contexts = result.get("contexts", [])

            st.markdown(answer)

            # Optional: Show contexts
            with st.expander("📄 Retrieved Context"):
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**Chunk {i+1}:** {ctx}")

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })