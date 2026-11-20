# 🧠 Modular RAG Knowledge Assistant

An end-to-end, modular **Retrieval-Augmented Generation (RAG)** pipeline and interactive **Streamlit** assistant designed to ingest documents, preprocess and chunk text, generate dense vector embeddings stored in **ChromaDB**, and deliver accurate, source-grounded answers via **OpenRouter LLMs**.

![Application Demo](assets/demo.png)

---

## ✨ Key Features
- **Modular 7-Stage Pipeline:** Clean separation of concerns from document ingestion to prompt construction for easy debugging and scaling.
- **Persistent Vector Search:** Uses **ChromaDB** for fast semantic similarity retrieval over custom knowledge bases.
- **Source-Grounded Answers:** Displays retrieved document titles and text chunks alongside every generated response to ensure transparency and reduce hallucinations.
- **Interactive Web Interface:** Built with **Streamlit** for real-time document question answering.

---

## 🏗️ System Architecture

    [01_documents.py] ──▶ [02_preprocessing.py] ──▶ [03_chunking.py]
                                                            │
    [07_prompting.py] ◀── [06_retrieve_context.py] ◀── [04_vector_representation.py]
            │                                               │
            ▼                                               ▼
    [streamlit_app.py] (Interactive UI)             [05_create_chroma_store.py]

### Module Breakdown
1. **`01_documents.py`**: Loads and parses raw source documents.
2. **`02_preprocessing.py`**: Cleans, normalizes, and formats extracted text.
3. **`03_chunking.py`**: Splits text into semantically coherent overlapping chunks.
4. **`04_vector_representation.py`**: Generates dense vector embeddings for each chunk.
5. **`05_create_chroma_store.py`**: Builds and persists the vector index in **ChromaDB**.
6. **`06_retrieve_context.py`**: Queries the vector store to retrieve top-k relevant context chunks.
7. **`07_prompting.py`**: Constructs context-augmented prompts and queries the LLM via **OpenRouter API**.
8. **`streamlit_app.py`**: Interactive frontend displaying answers and expandable retrieved sources.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Vector Database:** ChromaDB
- **LLM Integration:** OpenRouter API
- **Frontend:** Streamlit

---

## 🚀 How to Run Locally

1. **Clone the repository:**

       git clone https://github.com/SalehA7mad/RAG-Pipeline.git
       cd RAG-Pipeline

2. **Create and activate a virtual environment:**

       python -m venv .venv
       # Windows:
       .venv\Scripts\activate

3. **Install dependencies:**

       pip install -r requirements.txt

4. **Configure API Keys:**
   Set your `OPENROUTER_API_KEY` (and optional `OPENROUTER_MODEL`) as environment variables or inside `.streamlit/secrets.toml`.

5. **Build the Vector Store & Launch:**

       python 05_create_chroma_store.py
       streamlit run streamlit_app.py