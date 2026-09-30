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
DOC_DIR  = BASE_DIR / "data" / "documents"
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

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

* { scrollbar-width: thin; scrollbar-color: #1e3a52 transparent; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e3a52; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2d5478; }

/* ── Animations ── */
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 20px rgba(99,102,241,.4),0 0 40px rgba(99,102,241,.15); }
    50%      { box-shadow: 0 0 30px rgba(139,92,246,.5),0 0 60px rgba(139,92,246,.2); }
}
@keyframes fade-up {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fade-in {
    from { opacity:0; } to { opacity:1; }
}
@keyframes shimmer {
    0%   { background-position:-400% 0; }
    100% { background-position: 400% 0; }
}
@keyframes blink {
    0%,100% { opacity:1; } 50% { opacity:.2; }
}
@keyframes float {
    0%,100% { transform:translateY(0);   }
    50%      { transform:translateY(-7px); }
}
@keyframes border-dance {
    0%,100% { border-color:rgba(99,102,241,.15); }
    50%      { border-color:rgba(139,92,246,.28); }
}

/* ── App Background ── */
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%,   rgba(99,102,241,.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 5%,   rgba(139,92,246,.10) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 100%, rgba(16,185,129,.06) 0%, transparent 50%),
        #05080f;
    color: #e2e8f0; min-height:100vh;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080e1a 0%,#060c16 100%) !important;
    border-right: 1px solid rgba(99,102,241,.15) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.1rem 1rem; }

/* ── Brand ── */
.brand {
    display:flex; align-items:center; gap:13px;
    padding:6px 4px 22px;
    border-bottom:1px solid rgba(99,102,241,.12);
    margin-bottom:4px;
}
.brand-orb {
    width:46px; height:46px; border-radius:14px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#4f46e5,#7c3aed);
    animation:pulse-glow 3s ease-in-out infinite;
    font-size:22px; position:relative; overflow:hidden;
}
.brand-orb::after {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(255,255,255,.15),transparent);
}
.brand-name { font-size:18px; font-weight:800; letter-spacing:-.3px; color:#f1f5f9; }
.brand-tag  { font-size:10px; color:#64748b; letter-spacing:1.2px; margin-top:2px; font-weight:600; }

/* ── Section Labels ── */
.sec-label {
    color:#475569; font-size:9.5px; font-weight:700;
    letter-spacing:1.5px; text-transform:uppercase;
    margin:20px 0 10px; display:flex; align-items:center; gap:7px;
}
.sec-label::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,.2),transparent);
}

/* ── Upload Zone ── */
div[data-testid="stFileUploader"] {
    border:1.5px dashed rgba(99,102,241,.3) !important;
    border-radius:14px !important;
    background:rgba(99,102,241,.04) !important;
    transition:all .2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color:rgba(99,102,241,.6) !important;
    background:rgba(99,102,241,.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius:10px !important;
    border:1px solid rgba(99,102,241,.25) !important;
    background:rgba(99,102,241,.1) !important;
    color:#c7d2fe !important;
    font-weight:600 !important;
    font-size:12px !important;
    transition:all .2s ease !important;
}
.stButton > button:hover {
    border-color:rgba(99,102,241,.6) !important;
    background:rgba(99,102,241,.22) !important;
    transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(99,102,241,.25) !important;
    color:#e0e7ff !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── Doc Cards ── */
.doc-card {
    display:flex; align-items:center; gap:10px;
    background:rgba(15,23,42,.6);
    border:1px solid rgba(99,102,241,.15);
    border-radius:12px; padding:9px 11px; margin-bottom:7px;
    transition:all .2s ease; animation:fade-in .3s ease-out;
}
.doc-card:hover {
    border-color:rgba(99,102,241,.4);
    background:rgba(99,102,241,.08);
    transform:translateX(3px);
}
.doc-icon-wrap {
    width:34px; height:34px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}
.doc-icon-pdf  { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.2); }
.doc-icon-docx { background:rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.2); }
.doc-icon-txt  { background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.2); }
.doc-name { font-size:11.5px; font-weight:600; color:#cbd5e1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.doc-meta { font-size:9.5px; color:#475569; margin-top:2px; }

/* ── System Status Card ── */
.sys-card {
    background:rgba(8,14,26,.7);
    border:1px solid rgba(99,102,241,.15);
    border-radius:14px; padding:13px 14px; margin-bottom:10px;
    animation:border-dance 4s ease-in-out infinite;
}
.sys-row {
    display:flex; justify-content:space-between; align-items:center;
    font-size:11.5px; padding:5px 0;
    border-bottom:1px solid rgba(255,255,255,.04);
}
.sys-row:last-child { border-bottom:none; padding-bottom:0; }
.sys-row span:first-child { color:#64748b; }
.sys-pill {
    display:inline-flex; align-items:center; gap:5px;
    padding:2px 8px; border-radius:99px; font-size:10px; font-weight:600;
}
.pill-green { background:rgba(16,185,129,.12);  color:#34d399; border:1px solid rgba(16,185,129,.2); }
.pill-amber { background:rgba(245,158,11,.12);  color:#fbbf24; border:1px solid rgba(245,158,11,.2); }
.pill-red   { background:rgba(239,68,68,.12);   color:#f87171; border:1px solid rgba(239,68,68,.2); }
.pill-blue  { background:rgba(99,102,241,.12);  color:#818cf8; border:1px solid rgba(99,102,241,.2); }
.live-dot {
    width:6px; height:6px; border-radius:50%;
    animation:blink 1.8s ease-in-out infinite;
}
.dot-green { background:#34d399; box-shadow:0 0 6px #34d399; }
.dot-amber { background:#fbbf24; box-shadow:0 0 6px #fbbf24; }
.dot-red   { background:#f87171; box-shadow:0 0 6px #f87171; }

/* ── Hero ── */
.hero {
    position:relative; overflow:hidden;
    border:1px solid rgba(99,102,241,.2); border-radius:24px;
    background:linear-gradient(135deg,rgba(15,23,42,.95),rgba(8,14,26,.95));
    padding:28px 32px 26px; margin-bottom:20px;
    box-shadow:0 25px 60px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);
    animation:fade-up .5s ease-out;
}
.hero::before {
    content:''; position:absolute; inset:0; pointer-events:none;
    background:linear-gradient(105deg,transparent 40%,rgba(99,102,241,.07) 50%,transparent 60%);
    background-size:300% 100%; animation:shimmer 8s linear infinite;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:7px;
    padding:5px 12px; border-radius:99px;
    background:rgba(99,102,241,.1); border:1px solid rgba(99,102,241,.25);
    color:#a5b4fc; font-size:10px; font-weight:700;
    letter-spacing:.8px; margin-bottom:14px;
}
.hero h1 {
    font-size:34px; font-weight:900; letter-spacing:-1.2px;
    background:linear-gradient(90deg,#f1f5f9 0%,#a5b4fc 50%,#c4b5fd 100%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
    line-height:1.15; margin-bottom:8px;
}
.hero-sub { color:#64748b; font-size:13.5px; line-height:1.55; }
.hero-orb {
    position:absolute; right:28px; top:50%; transform:translateY(-50%);
    width:72px; height:72px; border-radius:50%;
    background:radial-gradient(circle,rgba(99,102,241,.3),rgba(139,92,246,.15) 50%,transparent 70%);
    display:flex; align-items:center; justify-content:center;
    font-size:32px; animation:float 4s ease-in-out infinite;
}

/* ── Metric Cards ── */
.metric-card {
    background:rgba(10,17,30,.8);
    border:1px solid rgba(99,102,241,.12);
    border-radius:16px; padding:16px;
    transition:all .22s ease; animation:fade-up .5s ease-out;
    position:relative; overflow:hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0;
    height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,.3),transparent);
}
.metric-card:hover {
    transform:translateY(-4px);
    border-color:rgba(99,102,241,.35);
    box-shadow:0 12px 30px rgba(99,102,241,.15);
}
.metric-icon { font-size:22px; margin-bottom:8px; }
.metric-key  { color:#475569; font-size:10px; text-transform:uppercase; letter-spacing:1px; font-weight:700; margin-bottom:5px; }
.metric-val  { font-size:20px; font-weight:800; color:#e2e8f0; }
.metric-sub  { font-size:10px; color:#334155; margin-top:3px; }

/* ── Panels ── */
.panel {
    background:rgba(8,14,26,.85);
    border:1px solid rgba(99,102,241,.12);
    border-radius:20px; padding:20px;
    animation:fade-up .55s ease-out;
    position:relative; overflow:hidden;
}
.panel::before {
    content:''; position:absolute; top:0; left:0; right:0;
    height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,.25),transparent);
}
.panel-header {
    display:flex; align-items:center; gap:9px;
    font-weight:700; font-size:14px; color:#cbd5e1;
    margin-bottom:16px; padding-bottom:12px;
    border-bottom:1px solid rgba(99,102,241,.1);
}
.panel-header-icon {
    width:30px; height:30px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    background:rgba(99,102,241,.15); font-size:15px;
}

/* ── Pipeline ── */
.pipeline { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.p-step {
    flex:1; min-width:80px; text-align:center;
    padding:13px 6px; border-radius:12px;
    border:1px solid rgba(99,102,241,.15);
    background:rgba(15,23,42,.6);
    transition:all .2s ease;
}
.p-step:hover {
    transform:translateY(-3px);
    border-color:rgba(99,102,241,.5);
    background:rgba(99,102,241,.1);
    box-shadow:0 8px 20px rgba(99,102,241,.15);
}
.p-step-icon { font-size:18px; margin-bottom:5px; }
.p-step-name { font-size:10px; font-weight:700; color:#cbd5e1; }
.p-step-desc { font-size:9px; color:#475569; margin-top:2px; }
.p-arrow     { color:rgba(99,102,241,.4); font-size:14px; flex-shrink:0; }

/* ── Config ── */
.config-row {
    display:flex; align-items:flex-start; gap:10px;
    padding:9px 0; border-bottom:1px solid rgba(255,255,255,.04);
    font-size:12px;
}
.config-row:last-child { border-bottom:none; }
.config-key { color:#475569; min-width:72px; font-weight:600; padding-top:1px; }
.config-val { color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:11px; word-break:break-all; }

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background:transparent !important;
    border:none !important;
    animation:fade-up .3s ease-out;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] > div {
    border:1px solid rgba(99,102,241,.25) !important;
    border-radius:16px !important;
    background:rgba(10,17,30,.9) !important;
    transition:border-color .2s ease, box-shadow .2s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color:rgba(99,102,241,.55) !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.1) !important;
}
[data-testid="stChatInput"] textarea { color:#e2e8f0 !important; font-size:13.5px !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#334155 !important; }

textarea, input[type="text"] { border-radius:12px !important; }

[data-testid="stExpander"] {
    border:1px solid rgba(99,102,241,.15) !important;
    border-radius:14px !important;
    background:rgba(8,14,26,.5) !important;
}

/* ── Empty State ── */
.empty-state {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding:48px 20px; text-align:center;
    animation:fade-in .5s ease-out;
}
.empty-icon {
    width:72px; height:72px; border-radius:50%;
    background:radial-gradient(circle,rgba(99,102,241,.2),rgba(139,92,246,.1));
    display:flex; align-items:center; justify-content:center;
    font-size:30px; margin-bottom:16px;
    animation:float 4s ease-in-out infinite;
    border:1px solid rgba(99,102,241,.2);
}
.empty-title { font-size:16px; font-weight:700; color:#94a3b8; margin-bottom:7px; }
.empty-desc  { font-size:12px; color:#334155; line-height:1.6; max-width:300px; }
.suggestions { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:16px; }
.sug-pill {
    padding:7px 13px; border-radius:99px;
    background:rgba(99,102,241,.08); border:1px solid rgba(99,102,241,.2);
    color:#818cf8; font-size:11px; font-weight:500; transition:all .2s ease;
}
.sug-pill:hover {
    background:rgba(99,102,241,.18); border-color:rgba(99,102,241,.5);
    color:#c7d2fe; transform:translateY(-1px);
}

/* ── Source Cards ── */
.source-card {
    background:rgba(10,17,30,.8);
    border:1px solid rgba(99,102,241,.12);
    border-left:3px solid #4f46e5;
    border-radius:0 12px 12px 0;
    padding:11px 13px; margin:6px 0;
    transition:all .18s ease; font-size:11.5px;
}
.source-card:hover {
    border-left-color:#7c3aed;
    background:rgba(99,102,241,.07);
    transform:translateX(3px);
}
.source-title { font-weight:700; color:#a5b4fc; margin-bottom:3px; }
.source-meta  { color:#475569; font-size:10px; }
.score-track  { height:3px; background:rgba(255,255,255,.06); border-radius:99px; margin-top:7px; overflow:hidden; }
.score-fill   { height:100%; border-radius:99px; background:linear-gradient(90deg,#4f46e5,#7c3aed); }

/* ── Stale Banner ── */
.stale-banner {
    display:flex; align-items:center; gap:10px;
    background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.2);
    border-radius:12px; padding:10px 14px;
    font-size:12.5px; color:#fbbf24; margin-bottom:14px;
    animation:fade-in .3s ease-out;
}

/* ── Tips Panel ── */
.tip-row { font-size:12px; color:#475569; line-height:1.8; margin-bottom:7px; }
.tip-row:last-child { margin-bottom:0; }
.tip-row strong { color:#64748b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── HELPERS ───────────────────────────
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
    saved, skipped_empty, overwritten = [], [], []
    for f in files:
        name = Path(f.name).name
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
    doc_list  = list_documents()
    total     = len(doc_list)
    failed    = []
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
    texts      = [d["text"] for d in docs_meta]
    embeddings = get_embedding_model().encode(texts)
    get_vector_store().build(embeddings, docs_meta)
    st.session_state.index_doc_count = total
    st.session_state.index_stale     = False
    return len(docs_meta), failed

def ask_local_llm(question):
    store = get_vector_store()
    if not store.exists():
        return None, []
    q       = get_embedding_model().encode([question])[0]
    results = store.search(q, 5)
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']} | Chunk: {r['chunk_id']}]\n{r['text']}"
        for r in results
    )
    llm = LocalLLMClient(
        os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        os.getenv("LLM_MODEL",    "qwen2.5-3b-instruct"),
    )
    return llm.answer(question, context), results


# ─────────────────────────── SESSION STATE ───────────────────────────
for key, default in [
    ("messages",        []),
    ("index_doc_count", None),
    ("index_stale",     False),
    ("confirm_clear",   False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

current_docs = list_documents()
faiss_ready  = get_vector_store().exists()

if faiss_ready and st.session_state.index_doc_count is not None and \
        len(current_docs) != st.session_state.index_doc_count:
    st.session_state.index_stale = True


# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-orb">🧠</div>
      <div>
        <div class="brand-name">NOVA</div>
        <div class="brand-tag">OFFLINE · LOCAL · PRIVATE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader",
    )

    col_save, col_build = st.columns(2)
    with col_save:
        if st.button("＋ Save Files", use_container_width=True):
            if uploaded:
                saved, skipped_empty, overwritten = save_uploaded_files(uploaded)
                if saved:
                    st.session_state.index_stale = True
                    msg = f"✅ Saved {len(saved)} file(s)"
                    if overwritten:
                        msg += f" · {len(overwritten)} replaced"
                    st.success(msg)
                if skipped_empty:
                    st.warning(f"Skipped {len(skipped_empty)} empty file(s)")
                st.rerun()
            else:
                st.info("Select PDF, DOCX or TXT files first.")

    with col_build:
        if st.button("⚡ Build Index", use_container_width=True):
            if not current_docs:
                st.warning("Add documents first.")
            else:
                progress = st.progress(0.0, text="Initializing…")
                def _cb(i, total, name):
                    progress.progress(
                        i / max(total, 1),
                        text=f"📄 {name} ({i + 1}/{total})"
                    )
                count, failed = rebuild_index(progress_cb=_cb)
                progress.progress(1.0, text="✅ Complete")
                time.sleep(0.3)
                progress.empty()
                if count:
                    st.success(f"⚡ Indexed {count} chunks from {len(current_docs) - len(failed)} doc(s)")
                else:
                    st.warning("No extractable text found.")
                if failed:
                    with st.expander(f"⚠️ {len(failed)} file(s) failed"):
                        for name, err in failed:
                            st.caption(f"**{name}**: {err}")
                st.rerun()

    st.markdown('<div class="sec-label">🗂️ Documents</div>', unsafe_allow_html=True)
    if current_docs:
        for p in current_docs:
            ext      = p.suffix.lower()
            icon     = "📕" if ext == ".pdf" else "📘" if ext == ".docx" else "📄"
            icon_cls = "doc-icon-pdf" if ext == ".pdf" else "doc-icon-docx" if ext == ".docx" else "doc-icon-txt"
            sz       = p.stat().st_size / 1024
            sz_str   = f"{sz:.0f} KB" if sz >= 1 else f"{p.stat().st_size} B"
            c1, c2   = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-icon-wrap {icon_cls}">{icon}</div>'
                    f'<div style="min-width:0">'
                    f'<div class="doc-name" title="{p.name}">{p.name}</div>'
                    f'<div class="doc-meta">{sz_str} &nbsp;·&nbsp; {ext[1:].upper()}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑", key=f"del_{p.name}", help=f"Remove {p.name}"):
                    p.unlink(missing_ok=True)
                    st.session_state.index_stale = True
                    st.rerun()
    else:
        st.markdown(
            '<div style="text-align:center;padding:18px 0;color:#334155;font-size:12px">'
            '📂 No documents yet<br><span style="font-size:11px">Upload files above</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-label">⚙️ System Status</div>', unsafe_allow_html=True)

    if faiss_ready and st.session_state.index_stale:
        kb_cls, kb_dot_cls, kb_lbl = "pill-amber", "dot-amber", "Stale"
    elif faiss_ready:
        kb_cls, kb_dot_cls, kb_lbl = "pill-green", "dot-green", "Ready"
    else:
        kb_cls, kb_dot_cls, kb_lbl = "pill-red",   "dot-amber", "Empty"

    model_name = os.getenv("LLM_MODEL", "qwen2.5-3b-instruct")

    st.markdown(f"""
    <div class="sys-card">
      <div class="sys-row">
        <span>Knowledge Base</span>
        <span class="sys-pill {kb_cls}">
          <span class="live-dot {kb_dot_cls}"></span>{kb_lbl}
        </span>
      </div>
      <div class="sys-row">
        <span>LLM Engine</span>
        <span class="sys-pill pill-green">
          <span class="live-dot dot-green"></span>Local
        </span>
      </div>
      <div class="sys-row">
        <span>Internet</span>
        <span class="sys-pill pill-blue">Not Required</span>
      </div>
      <div class="sys-row">
        <span>Vector DB</span>
        <span class="sys-pill pill-blue">FAISS</span>
      </div>
      <div class="sys-row">
        <span>Model</span>
        <span style="color:#818cf8;font-size:10px;font-family:'JetBrains Mono',monospace">{model_name}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ This will delete all documents and the vector index. Are you sure?")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ Yes, Clear", use_container_width=True):
                for p in current_docs:
                    p.unlink(missing_ok=True)
                get_vector_store().clear()
                st.session_state.messages        = []
                st.session_state.index_doc_count = None
                st.session_state.index_stale     = False
                st.session_state.confirm_clear   = False
                st.rerun()
        with cc2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


# ─────────────────────────── MAIN ───────────────────────────
# Hero
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">
    <span class="live-dot dot-green" style="width:5px;height:5px"></span>
    FULLY OFFLINE &nbsp;·&nbsp; ZERO CLOUD &nbsp;·&nbsp; PRIVATE AI
  </div>
  <h1>NOVA Offline Assistant</h1>
  <p class="hero-sub">
    Ask questions across your private documents using semantic search &amp; a local LLM.<br>
    Your data never leaves your machine.
  </p>
  <div class="hero-orb">🧠</div>
</div>
""", unsafe_allow_html=True)

# Stale Warning
if faiss_ready and st.session_state.index_stale:
    st.markdown(
        '<div class="stale-banner">⚠️ &nbsp; Documents changed — click <strong>⚡ Build Index</strong> in the sidebar to refresh.</div>',
        unsafe_allow_html=True,
    )

# Metric Cards
kb_val     = "🟢 Ready" if (faiss_ready and not st.session_state.index_stale) \
             else ("🟠 Stale" if faiss_ready else "🔴 Empty")
llm_label  = os.getenv("LLM_MODEL", "qwen2.5-3b-instruct")

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-icon">📄</div>
      <div class="metric-key">Documents</div>
      <div class="metric-val">{len(current_docs)}</div>
      <div class="metric-sub">in knowledge base</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-icon">🗄️</div>
      <div class="metric-key">Vector Index</div>
      <div class="metric-val" style="font-size:15px">{kb_val}</div>
      <div class="metric-sub">FAISS powered</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-icon">🤖</div>
      <div class="metric-key">Language Model</div>
      <div class="metric-val" style="font-size:12px;font-family:'JetBrains Mono',monospace">{llm_label}</div>
      <div class="metric-sub">running locally</div>
    </div>""", unsafe_allow_html=True)
with mc4:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-icon">🔒</div>
      <div class="metric-key">Privacy</div>
      <div class="metric-val">100%</div>
      <div class="metric-sub">offline &amp; local</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.7, 1], gap="large")

# ── RIGHT ──
with right:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-header-icon">🔗</div>
        RAG Pipeline
      </div>
      <div class="pipeline">
        <div class="p-step"><div class="p-step-icon">📄</div><div class="p-step-name">Ingest</div><div class="p-step-desc">PDF·DOCX·TXT</div></div>
        <div class="p-arrow">›</div>
        <div class="p-step"><div class="p-step-icon">🔤</div><div class="p-step-name">Clean</div><div class="p-step-desc">NLP</div></div>
        <div class="p-arrow">›</div>
        <div class="p-step"><div class="p-step-icon">🧩</div><div class="p-step-name">Embed</div><div class="p-step-desc">Vectors</div></div>
        <div class="p-arrow">›</div>
        <div class="p-step"><div class="p-step-icon">🔎</div><div class="p-step-name">Retrieve</div><div class="p-step-desc">FAISS</div></div>
        <div class="p-arrow">›</div>
        <div class="p-step"><div class="p-step-icon">🤖</div><div class="p-step-name">Generate</div><div class="p-step-desc">Local LLM</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-header-icon">⚙️</div>
        Model Configuration
      </div>
      <div class="config-row">
        <span class="config-key">Provider</span>
        <span class="config-val">LM Studio</span>
      </div>
      <div class="config-row">
        <span class="config-key">Model</span>
        <span class="config-val">{os.getenv('LLM_MODEL', 'qwen2.5-3b-instruct')}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Endpoint</span>
        <span class="config-val">{os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Mode</span>
        <span class="config-val">Fully Offline</span>
      </div>
      <div class="config-row">
        <span class="config-key">Embeddings</span>
        <span class="config-val">sentence-transformers (local)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-header-icon">💡</div>
        Quick Tips
      </div>
      <div class="tip-row">📤 <strong>Upload</strong> your PDFs, DOCX or TXT files</div>
      <div class="tip-row">⚡ <strong>Build Index</strong> to vectorize documents</div>
      <div class="tip-row">💬 <strong>Ask questions</strong> in natural language</div>
      <div class="tip-row">🔒 <strong>Everything</strong> stays on your machine</div>
    </div>
    """, unsafe_allow_html=True)

# ── LEFT: Chat ──
with left:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-header-icon">💬</div>
        Document Chat
        <span style="margin-left:auto;font-size:10px;color:#334155;font-weight:500">Powered by local LLM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🧠</div>
          <div class="empty-title">Your Private AI is Ready</div>
          <div class="empty-desc">
            Upload documents, build the knowledge index, then ask anything.
            All processing is 100% local — no internet required.
          </div>
          <div class="suggestions">
            <div class="sug-pill">📄 Summarize my document</div>
            <div class="sug-pill">🔍 Find key insights</div>
            <div class="sug-pill">❓ Ask a question</div>
            <div class="sug-pill">📊 Compare sections</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sources(results):
        with st.expander(f"📚 Retrieved Sources ({len(results)})"):
            for s in results:
                pct = max(0, min(100, round(s["score"] * 100)))
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-title">📎 {s["source"]}</div>'
                    f'<div class="source-meta">Chunk #{s["chunk_id"]} &nbsp;·&nbsp; Similarity: {s["score"]:.3f}</div>'
                    f'<div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])

    question = st.chat_input("Ask anything about your documents…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                if not current_docs:
                    answer  = "⚠️ Please upload at least one document first, then build the knowledge index."
                    results = []
                elif not get_vector_store().exists():
                    answer  = "⚡ Please click **Build Index** in the sidebar to process your documents before asking questions."
                    results = []
                else:
                    if st.session_state.index_stale:
                        st.caption("⚠️ Index may be outdated — consider rebuilding for best accuracy.")
                    with st.spinner("Searching documents · Generating answer…"):
                        answer, results = ask_local_llm(question)

                st.markdown(answer)
                if results:
                    render_sources(results)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": results,
                })
            except Exception as e:
                msg = (
                    f"**⚠️ Local model connection failed.**\n\n"
                    f"```\n{e}\n```\n\n"
                    "**Fix:** Open LM Studio → Local Model API → Start Server, "
                    "and verify the model name in `.env` matches the loaded model."
                )
                st.error(msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": msg, "sources": []
                })

# Footer
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;color:#1e293b;font-size:10px;letter-spacing:1px">
  NOVA OFFLINE AI &nbsp;·&nbsp; RAG + FAISS + LOCAL LLM &nbsp;·&nbsp; 🔒 Your data stays on your machine
</div>
""", unsafe_allow_html=True)
