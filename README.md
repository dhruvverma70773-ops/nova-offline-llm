# Offline NLP AI Assistant — Local Qwen

This version removes the Kimi API key. It uses a local Qwen model served by LM Studio.

## Setup
1. Install LM Studio.
2. Download a small Qwen Instruct model suitable for your PC.
3. Load the model and start the LM Studio local server.
4. In VS Code, open this folder.
5. Run:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
6. Copy `.env.example` to `.env`.
7. Set `LLM_MODEL` to the exact model identifier shown by LM Studio.
8. Run:
```powershell
streamlit run app.py
```

## RAG pipeline
Document → extraction → NLP cleaning → chunking → embeddings → FAISS → relevant chunks → local Qwen → answer.

The first-time package/model downloads require internet. After setup, the assistant can run locally without internet.
