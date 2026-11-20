from importlib import import_module
import streamlit as st

rag = import_module("07_prompting")

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🧠")
st.title("🧠 RAG Knowledge Assistant")

question = st.text_area("Ask a question about your documents:", placeholder="Type your question here...")

if st.button("Answer", type="primary") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            answer, sources = rag.answer_question(question)
            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("📚 Retrieved Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}. {source['title']}**")
                    st.write(source["chunk_text"])
                    st.divider()
        except Exception as e:
            st.error(f"An error occurred: {e}")