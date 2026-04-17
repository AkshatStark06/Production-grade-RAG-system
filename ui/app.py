import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from main import build_rag_pipeline
import requests

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
#@st.cache_resource
#def load_pipeline():
#    return build_rag_pipeline()

#rag = load_pipeline()
API_URL = "http://127.0.0.1:8000/query"


# ------------------------------
# SIDEBAR (NEW 🔥)
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    show_context = st.toggle("Show Retrieved Context", value=True)
    show_debug = st.toggle("Show Debug Info", value=False)


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

            try:
                # Run RAG pipeline
                response = requests.post(
                    API_URL,
                    json={"query": query}
                )

                if response.status_code == 200:
                    result = response.json()
                else:
                    st.error("API Error")
                    result = {"answer": "Error", "context": []}

                answer = result["answer"]
                contexts = result.get("context", [])

                # ------------------------------
                # CLEAN DISPLAY (IMPROVED)
                # ------------------------------
                st.markdown("### 🤖 Answer")
                st.markdown(answer)

                # ------------------------------
                # CONTEXT DISPLAY (IMPROVED)
                # ------------------------------
                if show_context:
                    with st.expander("📄 Retrieved Context"):
                        for i, ctx in enumerate(contexts, 1):

                            # Limit length for readability
                            MAX_CHARS = 300
                            content = (
                                ctx[:MAX_CHARS] + "..."
                                if len(ctx) > MAX_CHARS
                                else ctx
                            )

                            st.markdown(f"**Chunk {i}**")
                            st.markdown(content)
                            st.divider()

                # ------------------------------
                # DEBUG INFO (🔥 PORTFOLIO BOOST)
                # ------------------------------
                if show_debug:
                    with st.expander("🔍 Debug Info"):
                        st.json(result)

            except Exception as e:
                st.error("Something went wrong. Please try again.")
                answer = "Error occurred."

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })