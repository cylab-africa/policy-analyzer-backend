"""Keyword (BM25) search must weight rare words above common ones, which needs Qdrant's IDF
modifier on the collection's sparse vectors. Runs against an in-memory Qdrant."""
from qdrant_client import QdrantClient, models

from app.services import vector_store as vs

NAME = "docs"
SPARSE = NAME + "_sparse"
COMMON, RARE = 1, 2  # word ids: say "data" and "Rwanda"


def store_on(client):
    store = object.__new__(vs.QdrantVectorStore)
    store._client = client
    store._collection_name = NAME
    store._embedding_dimension = 4
    return store


def modifier(client):
    return client.get_collection(NAME).config.params.sparse_vectors[SPARSE].modifier


def add_chunks(client):
    """Twenty chunks repeat the common word; one names the rare word once."""
    vector = lambda indices, values: {SPARSE: models.SparseVector(indices=indices, values=values)}
    points = [models.PointStruct(id=i, vector=vector([COMMON], [3.0]), payload={"rare": False}) for i in range(20)]
    points.append(models.PointStruct(id=20, vector=vector([COMMON, RARE], [1.0, 1.0]), payload={"rare": True}))
    client.upsert(NAME, points)


def top_result_has_the_rare_word(client):
    query = models.SparseVector(indices=[COMMON, RARE], values=[1.0, 1.0])
    return client.query_points(NAME, query=query, using=SPARSE, limit=1).points[0].payload["rare"]


def create_without_idf(client):
    """A collection as earlier versions created it."""
    client.create_collection(
        NAME,
        vectors_config={NAME + "_dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
    )


def test_a_new_collection_weights_rare_words():
    client = QdrantClient(":memory:")
    store_on(client)._ensure_collection_exists()
    assert modifier(client) == models.Modifier.IDF
    add_chunks(client)
    assert top_result_has_the_rare_word(client)


def test_an_existing_collection_is_switched_to_idf_without_reindexing():
    client = QdrantClient(":memory:")
    create_without_idf(client)
    add_chunks(client)
    assert not top_result_has_the_rare_word(client)  # the problem being fixed

    store_on(client)._ensure_collection_exists()
    assert modifier(client) == models.Modifier.IDF
    assert top_result_has_the_rare_word(client)
    assert client.count(NAME).count == 21  # nothing was re-indexed or lost


def test_a_collection_already_using_idf_is_left_alone(monkeypatch):
    client = QdrantClient(":memory:")
    store_on(client)._ensure_collection_exists()
    updates = []
    monkeypatch.setattr(client, "update_collection", lambda *a, **k: updates.append(k))
    store_on(client)._ensure_collection_exists()
    assert updates == []


def test_a_failed_update_does_not_stop_the_backend(monkeypatch, capsys):
    client = QdrantClient(":memory:")
    create_without_idf(client)

    def refuse(*args, **kwargs):
        raise RuntimeError("this Qdrant version does not support modifiers")

    monkeypatch.setattr(client, "update_collection", refuse)
    store_on(client)._ensure_collection_exists()
    assert "could not turn on IDF" in capsys.readouterr().out
