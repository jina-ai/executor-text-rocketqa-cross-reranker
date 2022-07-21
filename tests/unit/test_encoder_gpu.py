import pytest
from executor import RocketQACrossEncoder
from docarray import Document, DocumentArray


@pytest.fixture(scope="session")
def basic_encoder() -> RocketQACrossEncoder:
    """
    reranker
    :return: cross encoder
    """
    return RocketQACrossEncoder()


def test_no_document(basic_encoder: RocketQACrossEncoder):
    """
    none
    :param basic_encoder: encoder
    """
    basic_encoder.rerank(None)


def test_empty_document(basic_encoder: RocketQACrossEncoder):
    """
    If the documentArray is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([])
    basic_encoder.rerank(docs)
    assert len(docs) == 0


def test_no_text_document(basic_encoder: RocketQACrossEncoder):
    """
    If the text property of document is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([Document()])
    basic_encoder.rerank(docs)
    assert len(docs) == 1
    assert len(docs[0].matches) == 0


@pytest.mark.parametrize('m_nums', [1, 3, 5])
def test_equals_to_k(basic_encoder: RocketQACrossEncoder, m_nums: int):
    """
    If the length of matches is greater than ground k
    then the value of k will be assigned to ground k,
    otherwise, the length of matches will be assigned to k
    :param basic_encoder: encoder
    :param m_nums: the length of matches
    """
    docs = DocumentArray([Document(text="曾经沧海难为水")])
    docs[0].matches = DocumentArray([Document(text="除却巫山不是云") for _ in range(m_nums)])
    print(len(docs[0].matches))
    basic_encoder.rerank(docs)
    if m_nums > 3:
        assert len(docs) == 1
        assert len(docs[0].matches) == 3
    else:
        assert len(docs) == 1
        assert len(docs[0].matches) == m_nums


def test_no_match(basic_encoder: RocketQACrossEncoder):
    """
    no "matches"
    :param basic_encoder: encoder
    """
    docs = DocumentArray([Document(text="襟三江而带五湖")])
    basic_encoder.rerank(docs)
    assert len(docs) == 1
    assert len(docs[0].matches) == 0


def test_rank_quality(basic_encoder: RocketQACrossEncoder):
    """
    Test the quality of the reranker
    :param basic_encoder :encoder
    """
    docs = DocumentArray([Document(text="是离愁，别是一番滋味在心头")])
    docs[0].matches = DocumentArray(
        [
            Document(id='A', text='剪不断理还乱'),
            Document(id='B', text='庭有枇杷树，吾妻死之年所手植也，今已亭亭如盖矣。'),
            Document(id='C', text='晴空一鹤排云上，便引诗情到碧霄。'),
        ]
    )
    basic_encoder.rerank(docs)
    matched = ['A', 'B', 'C']
    for i, match_id in enumerate(matched):
        assert docs[0].matches[i].id == match_id
