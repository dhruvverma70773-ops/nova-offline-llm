import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.document_loader import extract_text
from utils.text_processor import clean_text, split_text
from utils.embedding import EmbeddingModel
from utils.vector_store import VectorStore
from utils.local_llm import LocalLLMClient

load_dotenv()

BASE_DIR = Path(__file__).parent
DOC_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "index"
DOC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}

st.set_page_config(
    page_title="NOVA • Offline AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

* { scrollbar-width: thin; scrollbar-color: #2a4360 transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #2a4360; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #3d5f85; }

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(53, 122, 255, .13), transparent 30%),
        radial-gradient(circle at 85% 15%, rgba(123, 77, 255, .12), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(46, 214, 165, .05), transparent 40%),
        #060d17;
    color: #e8eef7;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091522 0%, #070f1a 100%);
    border-right: 1px solid #1d3045;
}
[data-testid="stSidebar"] > div:first-child { padding: 1rem .9rem; }

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 25px rgba(76,115,255,.28); }
    50% { box-shadow: 0 0 38px rgba(139,92,246,.42); }
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes blink-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: .35; }
}

.brand {
    display:flex; align-items:center; gap:12px; margin:4px 4px 22px;
}
.brand-icon {
    width:43px; height:43px; border-radius:13px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#286cff,#8b5cf6);
    animation: pulse-glow 3.5s ease-in-out infinite;
    font-size:22px;
}
.brand-title { font-size:18px; font-weight:800; letter-spacing:.2px; }
.brand-sub { color:#75879c; font-size:11px; margin-top:2px; letter-spacing:.5px; }

.section-label {
    color:#71859b; font-size:10px; font-weight:800;
    letter-spacing:1.2px; text-transform:uppercase;
    margin:22px 2px 9px;
    display:flex; align-items:center; gap:6px;
}

.status-card {
    border:1px solid #20364d; border-radius:14px;
    background:#0b1826; padding:12px; margin-bottom:9px;
}
.status-row {
    display:flex; justify-content:space-between; align-items:center;
    font-size:12px; margin:8px 0;
}
.dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:7px; background:#36d67a; box-shadow:0 0 9px #36d67a;
    animation: blink-dot 2.2s ease-in-out infinite;
}
.dot-off { background:#f2a93b; box-shadow:0 0 8px #f2a93b; }
.dot-stale { background:#ff5e7a; box-shadow:0 0 8px #ff5e7a; }

.main-wrap { max-width:1180px; margin:auto; }

.hero {
    position:relative; overflow:hidden;
    border:1px solid #1e3550; border-radius:22px;
    background:linear-gradient(135deg,rgba(17,34,55,.92),rgba(10,22,36,.92));
    padding:26px 28px; margin-bottom:18px;
    box-shadow:0 18px 55px rgba(0,0,0,.25);
    animation: fade-up .5s ease-out;
}
.hero::before {
    content:""; position:absolute; inset:0;
    background:linear-gradient(90deg, transparent, rgba(76,115,255,.08), transparent);
    background-size: 200% 100%;
    animation: shimmer 6s linear infinite;
    pointer-events:none;
}
.hero h1 {
    margin:0; font-size:32px; letter-spacing:-1px;
    background:linear-gradient(90deg,#e8eef7,#a9c3ff);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p { color:#8194aa; margin:8px 0 0; font-size:13px; }
.badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 10px; border-radius:999px;
    background:#10283d; border:1px solid #214c70;
    color:#71b9ff; font-size:10px; font-weight:700; margin-bottom:12px;
}
.badge .pulse {
    width:6px; height:6px; border-radius:50%; background:#36d67a;
    animation: blink-dot 1.8s ease-in-out infinite;
}

.metric {
    border:1px solid #1c344c; border-radius:15px;
    background:#0a1725; padding:15px;
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    animation: fade-up .5s ease-out;
}
.metric:hover {
    transform: translateY(-3px);
    border-color:#3d6ea8;
    box-shadow:0 10px 26px rgba(76,115,255,.16);
}
.metric-k { color:#71869c; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
.metric-v { font-size:21px; font-weight:800; margin-top:4px; }

.panel {
    border:1px solid #1c344c; border-radius:17px;
    background:rgba(8,20,33,.88); padding:17px;
    animation: fade-up .55s ease-out;
}
.panel-title { font-weight:700; font-size:14px; margin-bottom:12px; display:flex; align-items:center; gap:8px; }

.doc-item {
    display:flex; align-items:center; gap:10px;
    border:1px solid #1c344c; border-radius:11px;
    background:#0b1928; padding:10px; margin-bottom:8px;
    transition: border-color .15s ease, transform .15s ease;
}
.doc-item:hover { border-color:#3d6ea8; transform: translateX(2px); }
.doc-icon {
    width:32px; height:32px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    background:#172d45; font-size:15px; flex-shrink:0;
}
.doc-name { font-size:12px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.doc-meta { color:#657a90; font-size:10px; margin-top:2px; }

.source {
    border-left:3px solid #4d8dff; background:#0c1b2b;
    padding:10px 12px; border-radius:0 9px 9px 0; margin:7px 0;
    font-size:11px; transition: border-color .15s ease, background .15s ease;
}
.source:hover { border-left-color:#8b5cf6; background:#0e2033; }
.score-bar-track { height:4px; border-radius:3px; background:#132a3f; margin-top:6px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#4d8dff,#8b5cf6); }

.pipeline { display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
.step {
    flex:1; min-width:90px; text-align:center;
    padding:12px 6px; border:1px solid #1c344c; border-radius:11px;
    background:#0b1928; color:#a9b8c8; font-size:10px;
    transition: transform .15s ease, border-color .15s ease;
}
.step:hover { transform: translateY(-2px); border-color:#4c8dff; }
.step b { display:block; color:#e8eef7; margin-top:4px; font-size:11px; }
.arrow { color:#3f6d9b; }

.stButton > button {
    border-radius:10px !important;
    border:1px solid #254764 !important;
    background:#0d2032 !important;
    color:#dce9f6 !important;
    font-weight:600 !important;
    transition: all .15s ease !important;
}
.stButton > button:hover {
    border-color:#4c8dff !important;
    background:#112b44 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(76,115,255,.18) !important;
}
.stButton > button:active { transform: translateY(0); }

div[data-testid="stFileUploader"] {
    border:1px dashed #29455f; border-radius:12px; background:#0a1725;
    transition: border-color .15s ease;
}
div[data-testid="stFileUploader"]:hover { border-color:#4c8dff; }
textarea, input { border-radius:12px !important; }
[data-testid="stChatMessage"] {
    background:#0a1725;
    border:1px solid #172e45;
    border-radius:14px;
    margin:7px 0;
    animation: fade-up .3s ease-out;
}
[data-testid="stExpander"] { border-radius:12px !important; border-color:#1c344c !important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()

@st.cache_resource
def get_vector_store():
    return VectorStore(INDEX_DIR)

def list_documents():
    return sorted(
        [p for p in DOC_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT],
        key=lambda x: x.name.lower(),
    )

def save_uploaded_files(files):
    """Save uploads, skipping empty files and reporting name collisions."""
    saved, skipped_empty, overwritten = [], [], []
    for f in files:
        name = Path(f.name).name  # strip any path components for safety
        data = f.getbuffer()
        if len(data) == 0:
            skipped_empty.append(name)
            continue
        dest = DOC_DIR / name
        if dest.exists():
            overwritten.append(name)
        dest.write_bytes(data)
        saved.append(name)
    return saved, skipped_empty, overwritten

def rebuild_index(progress_cb=None):
    docs_meta = []
    doc_list = list_documents()
    total = len(doc_list)
    failed = []
    for idx, path in enumerate(doc_list):
        if progress_cb:
            progress_cb(idx, total, path.name)
        try:
            raw = extract_text(path)
        except Exception as e:
            failed.append((path.name, str(e)))
            continue
        chunks = split_text(clean_text(raw))
        for i, chunk in enumerate(chunks):
            docs_meta.append({"text": chunk, "source": path.name, "chunk_id": i})
    if not docs_meta:
        return 0, failed
    texts = [d["text"] for d in docs_meta]
    embeddings = get_embedding_model().encode(texts)
    get_vector_store().build(embeddings, docs_meta)
    st.session_state.index_doc_count = total
    st.session_state.index_stale = False
    return len(docs_meta), failed

def ask_local_llm(question):
    store = get_vector_store()
    if not store.exists():
        return None, []
    q = get_embedding_model().encode([question])[0]
    results = store.search(q, 5)
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']} | Chunk: {r['chunk_id']}]\n{r['text']}"
        for r in results
    )
    llm = LocalLLMClient(
        os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        os.getenv("LLM_MODEL", "qwen2.5-3b-instruct"),
    )
    return llm.answer(question, context), results

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_doc_count" not in st.session_state:
    st.session_state.index_doc_count = None
if "index_stale" not in st.session_state:
    st.session_state.index_stale = False
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

current_docs = list_documents()
faiss_ready = get_vector_store().exists()
# Detect staleness: doc count changed since last successful index build.
if faiss_ready and st.session_state.index_doc_count is not None and len(current_docs) != st.session_state.index_doc_count:
    st.session_state.index_stale = True

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-title">NOVA Assistant</div>
        <div class="brand-sub">PRIVATE • LOCAL • RAG</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader",
    )
    col_save, col_build = st.columns(2)
    with col_save:
        if st.button("＋ Save", use_container_width=True):
            if uploaded:
                saved, skipped_empty, overwritten = save_uploaded_files(uploaded)
                if saved:
                    st.session_state.index_stale = True
                    msg = f"Saved {len(saved)} document(s)"
                    if overwritten:
                        msg += f" ({len(overwritten)} replaced)"
                    st.success(msg)
                if skipped_empty:
                    st.warning(f"Skipped {len(skipped_empty)} empty file(s)")
                st.rerun()
            else:
                st.info("Select a PDF, DOCX or TXT file first.")
    with col_build:
        if st.button("⚡ Build", use_container_width=True):
            if not current_docs:
                st.warning("Add documents first.")
            else:
                progress = st.progress(0.0, text="Starting…")
                def _cb(i, total, name):
                    progress.progress(i / max(total, 1), text=f"Processing {name} ({i+1}/{total})")
                count, failed = rebuild_index(progress_cb=_cb)
                progress.progress(1.0, text="Done")
                time.sleep(0.2)
                progress.empty()
                if count:
                    st.success(f"Indexed {count} chunks from {len(current_docs) - len(failed)} document(s)")
                else:
                    st.warning("No extractable text found in the supported documents.")
                if failed:
                    with st.expander(f"⚠️ {len(failed)} file(s) failed to process"):
                        for name, err in failed:
                            st.caption(f"**{name}**: {err}")
                st.rerun()

    st.markdown('<div class="section-label">🗂 Local Documents</div>', unsafe_allow_html=True)
    if current_docs:
        for p in current_docs:
            icon = "📕" if p.suffix.lower() == ".pdf" else "📘" if p.suffix.lower() == ".docx" else "📄"
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div class="doc-item"><div class="doc-icon">{icon}</div>'
                    f'<div><div class="doc-name" title="{p.name}">{p.name}</div>'
                    f'<div class="doc-meta">{p.stat().st_size/1024:.1f} KB • local</div></div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑", key=f"del_{p.name}", help=f"Remove {p.name}"):
                    p.unlink(missing_ok=True)
                    st.session_state.index_stale = True
                    st.rerun()
    else:
        st.caption("No documents added yet.")

    st.markdown('<div class="section-label">⚙️ System</div>', unsafe_allow_html=True)
    if faiss_ready and st.session_state.index_stale:
        kb_dot, kb_label = "dot-stale", "Stale — rebuild"
    elif faiss_ready:
        kb_dot, kb_label = "dot", "Ready"
    else:
        kb_dot, kb_label = "dot-off", "Empty"
    st.markdown(f"""
    <div class="status-card">
      <div class="status-row"><span>Knowledge Base</span><span><i class="dot {kb_dot}"></i>{kb_label}</span></div>
      <div class="status-row"><span>LLM</span><span><i class="dot"></i>Local</span></div>
      <div class="status-row"><span>Internet</span><span><i class="dot"></i>Not required</span></div>
      <div class="status-row"><span>Engine</span><span>FAISS</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.confirm_clear:
        if st.button("🗑 Clear Local Data", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This deletes all documents and the index. Are you sure?")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ Yes, clear", use_container_width=True):
                for p in current_docs:
                    p.unlink(missing_ok=True)
                get_vector_store().clear()
                st.session_state.messages = []
                st.session_state.index_doc_count = None
                st.session_state.index_stale = False
                st.session_state.confirm_clear = False
                st.rerun()
        with cc2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

# -------------------- MAIN --------------------
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge"><span class="pulse"></span> LOCAL AI • NO CLOUD REQUIRED</div>
  <h1>Offline NLP AI Assistant</h1>
  <p>Ask questions about your documents using semantic retrieval and a local Qwen language model.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric"><div class="metric-k">Documents</div><div class="metric-v">📄 {len(current_docs)}</div></div>', unsafe_allow_html=True)
with c2:
    kb_v = "🟢 Ready" if (faiss_ready and not st.session_state.index_stale) else ("🟠 Stale" if faiss_ready else "🟠 Empty")
    st.markdown(f'<div class="metric"><div class="metric-k">Knowledge Base</div><div class="metric-v">{kb_v}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric"><div class="metric-k">LLM</div><div class="metric-v">🤖 Qwen</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric"><div class="metric-k">Privacy</div><div class="metric-v">🔒 Local</div></div>', unsafe_allow_html=True)

if faiss_ready and st.session_state.index_stale:
    st.info("📌 Your documents changed since the last index build. Click **⚡ Build** in the sidebar to refresh answers.")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")

with right:
    st.markdown('<div class="panel"><div class="panel-title">🔗 RAG Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
      <div class="step">📄<b>Documents</b>Upload</div>
      <div class="arrow">→</div>
      <div class="step">🔤<b>NLP</b>Clean</div>
      <div class="arrow">→</div>
      <div class="step">🧩<b>Embeddings</b>Vectorize</div>
      <div class="arrow">→</div>
      <div class="step">🔎<b>FAISS</b>Retrieve</div>
      <div class="arrow">→</div>
      <div class="step">🤖<b>Qwen</b>Generate</div>
    </div></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    st.write("**Provider:** LM Studio")
    st.write(f"**Model:** `{os.getenv('LLM_MODEL', 'qwen2.5-3b-instruct')}`")
    st.write(f"**Endpoint:** `{os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')}`")
    st.caption("The model runs on your computer through LM Studio.")
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    st.markdown('<div class="panel"><div class="panel-title">💬 Document Chat</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div style="padding:32px 10px;text-align:center;color:#71869c">
          <div style="font-size:38px">🧠</div>
          <div style="font-size:15px;color:#cbd8e6;font-weight:700;margin-top:8px">Your local knowledge assistant</div>
          <div style="font-size:11px;margin-top:6px">Build the knowledge base, then ask anything about your documents.</div>
        </div>
        """, unsafe_allow_html=True)

    def render_sources(results):
        with st.expander(f"📚 Retrieved sources ({len(results)})"):
            for s in results:
                pct = max(0, min(100, round(s["score"] * 100)))
                st.markdown(
                    f'<div class="source"><b>{s["source"]}</b> • chunk {s["chunk_id"]}<br>'
                    f'<span style="color:#71869c">Similarity: {s["score"]:.3f}</span>'
                    f'<div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])

    question = st.chat_input("Ask something about your documents…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                if not current_docs:
                    answer, results = "Add at least one document before asking questions.", []
                elif not get_vector_store().exists():
                    answer, results = "Please build the knowledge base first using **⚡ Build** in the sidebar.", []
                else:
                    if st.session_state.index_stale:
                        st.caption("⚠️ Answering from a possibly outdated index — consider rebuilding.")
                    with st.spinner("Searching documents and asking local Qwen…"):
                        answer, results = ask_local_llm(question)
                st.markdown(answer)
                if results:
                    render_sources(results)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
            except Exception as e:
                msg = (
                    f"**Local model connection failed.**\n\n`{e}`\n\n"
                    "Start the LM Studio local server and make sure the model name in `.env` matches the loaded model."
                )
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#536a80;font-size:10px;margin:25px 0 5px">
  NOVA OFFLINE AI • RAG + NLP + FAISS + LOCAL QWEN • Your documents stay on your computer
</div>
""", unsafe_allow_html=True)
