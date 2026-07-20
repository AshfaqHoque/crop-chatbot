import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "crops"


def load_documents(path="data/documents.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents to embed")

    embeddings = OllamaEmbeddings(model="bge-m3")

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    # stable, human-readable ids: crop_id + section + position, so re-running
    # this script updates existing entries instead of duplicating them
    ids = [f"{m['crop_id']}-{m['section']}-{i}" for i, m in enumerate(metadatas)]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # embed + store in batches so we get progress feedback (683 docs is small,
    # but this habit matters once you have thousands)
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    print(f"\nDone. Index persisted to ./{PERSIST_DIR}")
    return vectorstore


def test_search(vectorstore, query: str, k: int = 3):
    print(f"\n--- search: '{query}' ---")
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    for doc, score in results:
        print(f"score={score:.3f} | crop={doc.metadata['crop_name']} | section={doc.metadata['section']}")
        print(f"  {doc.page_content[:150]}")


if __name__ == "__main__":
    vs = build_index()

    # sanity-check queries — pick ones relevant to what you actually want to ask later
    test_search(vs, "boro rice seed rate per hectare")
    test_search(vs, "ধানের বীজ হার")  # same question in Bangla
    test_search(vs, "pest control for rice")