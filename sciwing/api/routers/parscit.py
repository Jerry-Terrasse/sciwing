from fastapi import APIRouter
from sciwing.models.neural_parscit import NeuralParscit
from typing import List

router = APIRouter()

parscit_model = None


@router.post("/parscit/batch")
def tag_citation_batch(citations: List[str]):
    """ End point to tag parse a batch of reference strings to their constituent parts.

    Parameters
    ----------
    citations: List[str]
        The reference strings to be parsed.

    Returns
    -------
    JSON
        ``{"tags": Predicted tags, "text_tokens": Tokenized citation string}``

    """
    try:
        assert isinstance(citations, list)
        assert len(citations) > 0
        for citation in citations:
            assert isinstance(citation, str)
        assert ''.join(citations) != ''
    except AssertionError:
        return [{"tags": "", "text_tokens": []} for _ in citations] # return empty tags and tokens if input is invalid
    global parscit_model
    if parscit_model == None:
        parscit_model = NeuralParscit()
    predictions = parscit_model.predict_for_batch(citations, show=False)
    return [{"tags": pred, "text_tokens": citation.split()} for pred, citation in zip(predictions, citations)]

@router.get("/parscit/{citation}")
def tag_citation_string(citation: str):
    """ End point to tag parse a reference string to their constituent parts.

    Parameters
    ----------
    citation: str
        The reference string to be parsed.

    Returns
    -------
    JSON
        ``{"tags": Predicted tags, "text_tokens": Tokenized citation string}``

    """
    global parscit_model
    if parscit_model == None:
        parscit_model = NeuralParscit()
    predictions = parscit_model.predict_for_text(citation)
    return {"tags": predictions, "text_tokens": citation.split()}