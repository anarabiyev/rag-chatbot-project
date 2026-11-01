import os
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.paths import DB_PATH

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def client():
    return chromadb.PersistentClient(str(DB_PATH))


@lru_cache(maxsize=1)
def embedder():  # loading the model is the slow part, so every collection shares one
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, normalize_embeddings=True)


def get_collection(name, reset=False):
    if EMBED_MODEL != DEFAULT_MODEL:
        name += "_" + EMBED_MODEL.split("/")[-1]
    if reset:
        try:
            client().delete_collection(name)
        except Exception:  # collection did not exist yet
            pass
    return client().get_or_create_collection(name, embedding_function=embedder(), metadata={"hnsw:space": "cosine"})


def add_chunks(collection, chunks, batch=256):
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.add(
            ids=[f"c{start + i}" for i in range(len(part))],
            documents=[c["text"] for c in part],
            metadatas=[{k: v for k, v in c.items() if k != "text"} for c in part],
        )
        print(f"  {min(start + batch, len(chunks))}/{len(chunks)}", end="\r")
    print()
