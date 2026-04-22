import streamlit as st
import sys
import os
import requests
import time

# ------------------------------
# STREAMING FUNCTION
# ------------------------------
def stream_response(query):
    response = requests.post(
        STREAM_URL,
        json={"query": query},
        stream=True
    )

    for line in response.iter_lines(decode_unicode=True):
        if line:
            if line.startswith("data: "):
                yield line.replace("data: ", "")


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production-Grade RAG Chatbot")

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/query"
STREAM_URL = f"{BASE_URL}/query-stream"


# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    show_context = st.toggle("Show Retrieved Context", value=True)
    show_debug = st.toggle("Show Debug Info", value=False)
    default_stream = os.getenv("STREAMING_ENABLED", "true") == "true"
    use_streaming = st.toggle("Enable Streaming", value=default_stream)


# ------------------------------
# SESSION STATE
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

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:

                # ------------------------------
                # STREAMING MODE
                # ------------------------------
                if use_streaming:
                    placeholder = st.empty()
                    full_text = ""

                    for partial_text in stream_response(query):
                        full_text += partial_text  # ✅ FIX HERE
                        placeholder.markdown(f"### 🤖 Answer\n{full_text}")

                    answer = full_text
                    contexts = []

                # ------------------------------
                # NORMAL MODE
                # ------------------------------
                else:
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

                    st.markdown("### 🤖 Answer")
                    st.markdown(answer)

                # ------------------------------
                # CONTEXT DISPLAY
                # ------------------------------
                if show_context and not use_streaming:
                    with st.expander("📄 Retrieved Context"):
                        for i, ctx in enumerate(contexts, 1):

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
                # DEBUG INFO
                # ------------------------------
                if show_debug and not use_streaming:
                    with st.expander("🔍 Debug Info"):
                        st.json(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                answer = "Error occurred."

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })