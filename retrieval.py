import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"


def terms(text):
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    return re.findall(r"[\w]+", normalized)


class Encoder:
    def __init__(self, model_name=EMBEDDING_MODEL, revision=None):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)

        module_file = Path(__file__)
        model_cache = module_file.parent / "models" / "hub"
        cache_folder = str(model_cache)
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
            cache_folder=cache_folder,
            model_kwargs={"dtype": "float32"},
        )
        self.tokenizer = self.model.tokenizer

    def encode(self, texts):
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_count = len(tokens)
            if token_count > self.model.max_seq_length:
                raise ValueError("A source exceeds the embedding model's context window")
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class HybridIndex:
    def __init__(self, records, vectors, encoder):
        self.records = records
        self.encoder = encoder
        self.vectors = np.asarray(vectors, dtype=np.float32)
        token_lists = []
        self.token_sets = []
        self.content_sets = []
        count = len(records)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != count:
            raise ValueError("Index records and embedding rows do not match")
        lengths = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if not np.all(lengths > 0):
            raise ValueError("Index contains empty embeddings")
        self.vectors = self.vectors / lengths
        for record in records:
            searchable = record["source_name"] + "\n" + record["title"] + "\n" + record["text"]
            tokens = terms(searchable)
            token_set = set(tokens)
            content = terms(record["text"])
            content_set = set(content)
            token_lists.append(tokens)
            self.token_sets.append(token_set)
            self.content_sets.append(content_set)
        self.bm25 = BM25Okapi(token_lists) if records else None

    def search(self, query, limit=10):
        query_tokens = terms(query)
        if not query_tokens or not self.records:
            return []
        query_set = set(query_tokens)
        lexical_scores = self.bm25.get_scores(query_tokens)
        lexical_order = np.argsort(-lexical_scores, kind="stable")
        lexical = []
        for index in lexical_order:
            shared = self.token_sets[index] & query_set
            if shared:
                lexical.append(index)
        lexical = lexical[:20]
        query_vectors = self.encoder.encode([query])
        query_vector = query_vectors[0]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query produced an empty embedding")
        query_vector = query_vector / query_norm
        semantic_scores = self.vectors @ query_vector
        semantic_order = np.argsort(-semantic_scores, kind="stable")
        semantic = semantic_order[:20]
        fused = {}
        for ranking in (lexical, semantic):
            for rank, index in enumerate(ranking, start=1):
                previous = fused.get(index, 0)
                fused[index] = previous + 1 / (60 + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)
        selected = []
        results = []
        for index in ordered:
            record = self.records[index]
            duplicate = False
            for previous in selected:
                previous_record = self.records[previous]
                if record["source_name"] != previous_record["source_name"]:
                    continue
                record_pages = record.get("pages", [])
                previous_pages = previous_record.get("pages", [])
                if record_pages != previous_pages:
                    continue
                shared = self.content_sets[index] & self.content_sets[previous]
                combined = self.content_sets[index] | self.content_sets[previous]
                shared_count = len(shared)
                combined_count = len(combined)
                if combined_count and shared_count / combined_count > 0.85:
                    duplicate = True
                    break
            if duplicate:
                continue
            results.append(record)
            selected.append(index)
            count = len(results)
            if count >= limit:
                break
        return results
