from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
