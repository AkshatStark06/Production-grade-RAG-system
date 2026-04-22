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

st.markdown("### 💡 Try these example questions")

st.info("""
• What is this system?  
• How does the retrieval process work?  
• What models are used in this RAG pipeline?  
• Explain the role of reranking  
""")

with st.expander("📄 About the Knowledge Base"):
    st.markdown("""
This chatbot is powered by a curated dataset containing information about:

- Retrieval-Augmented Generation (RAG)
- Vector databases (FAISS)
- Embedding models
- Reranking techniques

Try asking questions related to these topics for best results.
""")

with st.expander("📂 View Sample Knowledge Base (sample.txt)"):
    try:
        with open("data/sample.txt", "r", encoding="utf-8") as f:
            content = f.read()

        MAX_CHARS = 2000  # limit display size
        display_text = content[:MAX_CHARS]

        st.text_area(
            "Sample Data Preview",
            display_text,
            height=300
        )
        with open("data/sample.txt", "rb") as f:
            st.download_button(
                label="⬇️ Download Full Sample File",
                data=f,
                file_name="sample.txt",
                mime="text/plain"
            )

        if len(content) > MAX_CHARS:
            st.caption("Showing partial content...")

    except Exception as e:
        st.error(f"Could not load sample.txt: {e}")

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/query"
STREAM_URL = f"{BASE_URL}/query-stream"


# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # ------------------------------
    # MAIN MODE SELECTION
    # ------------------------------
    mode = st.radio(
        "Select Mode",
        ["Streaming Mode", "Debug Mode"]
    )
    st.caption(" Streaming = faster answers | Debug = detailed insights")

    # ------------------------------
    # DEFAULT FLAGS
    # ------------------------------
    use_streaming = False
    show_context = False
    show_debug = False

    # ------------------------------
    # STREAMING MODE
    # ------------------------------
    if mode == "Streaming Mode":
        use_streaming = True
        st.success("⚡ Fast response mode (no context/debug)")

    # ------------------------------
    # DEBUG MODE
    # ------------------------------
    else:
        st.info(" Debug Mode: Select what to display")

        show_context = st.checkbox("📄 Show Retrieved Context", value=True)
        show_debug = st.checkbox("🔍 Show Debug Info", value=False)

        # Optional UX feedback
        if show_context and show_debug:
            st.success("Showing: Context + Debug")
        elif show_context:
            st.success("Showing: Context only")
        elif show_debug:
            st.success("Showing: Debug only")
        else:
            st.warning("Nothing selected — only answer will be shown")


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
                    #st.write("DEBUG contexts:", contexts)
                    confidence = result.get("confidence", None)

                    st.markdown("### 🤖 Answer")
                    st.markdown(answer)
                    if confidence is not None:
                        if confidence > 0.85:
                            st.success(f"🟢 High Confidence ({confidence:.2f})")
                        elif confidence > 0.6:
                            st.warning(f"🟡 Medium Confidence ({confidence:.2f})")
                        else:
                            st.error(f"🔴 Low Confidence ({confidence:.2f})")
                    st.caption("Confidence reflects how strongly the retrieved context matches your query (based on reranker score).")

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