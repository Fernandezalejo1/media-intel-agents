"""Retrieval (RAG) tests: embeddings, vector search, retriever."""

from __future__ import annotations

import pytest

from media_intel.data.retrieval import LocalVectorSearch, MockEmbeddings, Retriever


def test_mock_embeddings_deterministic_and_normalized():
    e = MockEmbeddings(dim=64)
    a = e.embed(["hello world"])[0]
    b = e.embed(["hello world"])[0]
    assert a == b
    assert len(a) == 64
    assert abs(sum(x * x for x in a) - 1.0) < 1e-3


def test_vector_search_upsert_and_query():
    e = MockEmbeddings(dim=64)
    vs = LocalVectorSearch(root=".tmp-test/vectors", embedding_dim=64)
    texts = ["vinicius goal", "audience spike", "editorial policy"]
    vecs = e.embed(texts)
    records = [{"id": f"d{i}", "text": t, "vector": v, "metadata": {"topic": "sports"}} for i, (t, v) in enumerate(zip(texts, vecs))]
    vs.upsert("test-index", records)

    q = e.embed(["goal scored by winger"])[0]
    hits = vs.query("test-index", q, k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "vinicius goal"
    assert 0.0 <= hits[0]["score"] <= 1.0001


def test_vector_search_metadata_filter():
    e = MockEmbeddings(dim=64)
    vs = LocalVectorSearch(root=".tmp-test/vectors", embedding_dim=64)
    texts = ["goal clip", "policy doc"]
    vecs = e.embed(texts)
    records = [
        {"id": "a", "text": texts[0], "vector": vecs[0], "metadata": {"kind": "clip"}},
        {"id": "b", "text": texts[1], "vector": vecs[1], "metadata": {"kind": "doc"}},
    ]
    vs.upsert("filtered", records)
    q = e.embed(["goal"])[0]
    hits = vs.query("filtered", q, k=5, filter={"kind": "doc"})
    assert [h["id"] for h in hits] == ["b"]


def test_vector_dim_mismatch_rejected():
    vs = LocalVectorSearch(root=".tmp-test/vectors", embedding_dim=8)
    with pytest.raises(ValueError, match="dim"):
        vs.upsert("bad", [{"id": "x", "text": "t", "vector": [0.0] * 16, "metadata": {}}])


def test_retriever_end_to_end_rag_query():
    e = MockEmbeddings(dim=64)
    vs = LocalVectorSearch(root=".tmp-test/vectors", embedding_dim=64)
    docs = ["Bellingham scored the winner", "house style: lead with turning point"]
    vecs = e.embed(docs)
    vs.upsert("kb", [{"id": str(i), "text": t, "vector": v, "metadata": {}} for i, (t, v) in enumerate(zip(docs, vecs))])
    retriever = Retriever(e, vs)
    hits = retriever.retrieve("kb", "who scored the winner", k=1)
    assert hits and "Bellingham" in hits[0]["text"]
