from unittest.mock import Mock
import uuid

import numpy as np
import pytest

from database.qdrant_client import QdrantClientWrapper
from pipeline.ingest_pipeline import IngestPipelineWorker
from rag.retriever import LegalHybridRetriever
from rag.generator import LegalGenerator
from qdrant_client import QdrantClient


@pytest.mark.parametrize('detail,api,expected', [
    ({'effStatus': 1}, {}, 'Còn hiệu lực'),
    ({}, {'effStatus': 0}, 'Hết hiệu lực'),
    ({'effStatus': {'name': 'Hết hiệu lực một phần'}}, {}, 'Hết hiệu lực một phần'),
    ({'effStatus': {}}, {}, 'Chưa xác định'),
    ({}, {}, 'Chưa xác định'),
])
def test_merge_status_handles_numeric_and_missing_metadata(detail, api, expected):
    assert IngestPipelineWorker._parse_safe_status({}, detail, api) == expected


def test_merge_raw_es_client_receives_valid_query_and_effectiveness_filter():
    es = Mock(spec=['search'])
    es.search.return_value = {'hits': {'hits': [
        {'_id': 'article-1', '_score': 2.0, '_source': {'content': 'source'}},
    ]}}
    retriever = LegalHybridRetriever(qdrant_wrapper=Mock(), es_client=es,
        encoder_model=Mock(), graph_embeddings_cache={})
    assert retriever._search_sparse_es('Điều 1', top_k=3)[0]['chunk_id'] == 'article-1'
    arguments = es.search.call_args.kwargs
    assert arguments['size'] == 3
    assert arguments['query']['bool']['filter'] == [{'term': {'is_effective': True}}]
    assert 'query' not in arguments['query']
    assert arguments['query']['bool']['must'][0]['multi_match']['type'] == 'best_fields'


def test_merge_dense_ids_are_stable_and_distinct_and_vectors_accept_numpy():
    wrapper = QdrantClientWrapper.__new__(QdrantClientWrapper)
    wrapper.client = Mock()
    wrapper.vector_dim = 2
    wrapper.default_collection = 'test'
    records = [dict(chunk_id=cid, doc_id='d1', embedding=np.array([1., 2., 3.]),
                    content='source', is_effective=False)
               for cid in ['d1_art_1', 'd1_art_1_occurrence_2']]
    assert wrapper.upsert_batch(records) == 2
    first = wrapper.client.upsert.call_args.kwargs['points']
    assert len({p.id for p in first}) == 2
    assert all(str(uuid.UUID(p.id)) == p.id for p in first)
    assert all(p.vector == [1., 2.] and p.payload['is_effective'] is False for p in first)
    wrapper.upsert_batch(records)
    assert [p.id for p in first] == [p.id for p in wrapper.client.upsert.call_args.kwargs['points']]


def test_merge_attribution_does_not_match_article_one_inside_article_ten():
    generator = LegalGenerator.__new__(LegalGenerator)
    assert generator._compute_attribution_score('Theo Điều 10.',
        [{'doc_number': 'N/A', 'hierarchy_path': 'Điều 1'}]) == 0.0


def test_merge_qdrant_local_upsert_resume_and_effective_search():
    wrapper = QdrantClientWrapper.__new__(QdrantClientWrapper)
    wrapper.client = QdrantClient(location=':memory:')
    wrapper.vector_dim = 2
    wrapper.default_collection = 'merge-test'
    try:
        wrapper.init_collection()
        records = [dict(chunk_id=cid, doc_id='d1', embedding=np.array([1., 0., 9.]),
                        content=cid, is_effective=effective)
                   for cid, effective in [('d1_art_1', True), ('d1_art_1_occurrence_2', False)]]
        assert wrapper.upsert_batch(records) == 2
        assert wrapper.doc_exists('d1')
        assert not wrapper.doc_exists('missing')
        assert wrapper.upsert_batch(records) == 2
        assert wrapper.count_points() == 2
        assert [hit['chunk_id'] for hit in wrapper.search_dense(np.array([1., 0., 9.]))] == ['d1_art_1']
        assert len(wrapper.search_dense([1., 0.], must_be_effective=False)) == 2
    finally:
        wrapper.close()
