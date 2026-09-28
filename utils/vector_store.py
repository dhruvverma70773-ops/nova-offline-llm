import json
from pathlib import Path
import faiss, numpy as np

class VectorStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory/"faiss.index"
        self.meta_path = self.directory/"metadata.json"
    def exists(self): return self.index_path.exists() and self.meta_path.exists()
    def build(self, embeddings, documents):
        x = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(x.shape[1]); index.add(x)
        faiss.write_index(index, str(self.index_path))
        self.meta_path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    def search(self, query_embedding, top_k=5):
        if not self.exists():
            return []
        index = faiss.read_index(str(self.index_path))
        if index.ntotal == 0:
            return []
        docs = json.loads(self.meta_path.read_text(encoding="utf-8"))
        scores, ids = index.search(np.asarray([query_embedding], dtype="float32"), min(top_k, index.ntotal))
        return [{**docs[int(i)], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]
    def clear(self):
        for p in (self.index_path,self.meta_path):
            if p.exists(): p.unlink()
