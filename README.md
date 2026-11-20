# 🧠 Local RAG Knowledge Assistant (Modular Pipeline)

An end-to-end, modular **Retrieval-Augmented Generation (RAG)** pipeline and interactive **Streamlit** assistant built to ingest documents, preprocess text, generate vector embeddings, store them in **ChromaDB**, and answer context-grounded queries accurately.

![Demo Preview](assets/demo.png)

---

## 🏗️ Pipeline Architecture

The system is decoupled into 7 clear, maintainable stages:

```text
[01_documents.py] ──▶ [02_preprocessing.py] ──▶ [03_chunking.py]
                                                        │
[07_prompting.py] ◀── [06_retrieve_context.py] ◀── [04 & 05 Chroma Vector Store]
        │
        ▼
[streamlit_app.py] (Interactive Q&A Interface)