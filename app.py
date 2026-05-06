import streamlit as st
import requests

st.set_page_config(page_title="IndiaDoc AI", layout="wide")

# 🔥 Custom CSS (Premium look)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.stTextInput>div>div>input {
    font-size: 16px;
    padding: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# 🔥 Sidebar
with st.sidebar:
    st.title("🇮🇳 IndiaDoc AI")
    st.markdown("### ⚡ Legal Intelligence System")

    st.markdown("---")

    doc_option = st.selectbox(
        "📂 Select Document",
        ["All", "dpdp.pdf", "it_act.pdf","rbi_guidelines.pdf"]
    )

    selected_doc = None if doc_option == "All" else doc_option

    st.markdown("---")
    st.markdown("### 🧠 Features")
    st.markdown("""
    - Multi-document RAG  
    - Legal-aware retrieval  
    - Source citations  
    - Document filtering  
    """)

# 🔥 Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔥 Title
st.markdown("## 💬 Ask Legal Questions")

# 🔥 Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 🔥 Chat input
question = st.chat_input("Ask about laws, penalties, compliance...")

if question:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents... ⚡"):

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={
                        "question": question,
                        "document": selected_doc
                    }
                )

                data = response.json()
                answer = data["answer"]

                st.write(answer)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # 🔥 Sources section
                st.markdown("### 📚 Sources")

                cols = st.columns(2)

                for i, src in enumerate(data["sources"]):
                    with cols[i % 2]:
                        with st.container():
                            st.markdown(f"""
                            **📄 {src['document']}**  
                            Page: {src['page']}
                            """)
                            st.caption(src["content"])

            except Exception as e:
                st.error("Error connecting to API")