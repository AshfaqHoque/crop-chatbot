"""
Pydantic models mirroring the JSON shape of your existing context
retrieval API (the one that returns `contexts`, `context_text`,
`retrieval_confidence`, etc).

Keeping these separate from the client itself means: if the API's
response shape changes, you update this one file, and every caller
downstream (which only ever touches `ContextAPIResponse` objects, never
raw dicts) is unaffected as long as the fields they use still exist.
"""

from pydantic import BaseModel, Field


class DetectedEntity(BaseModel):
    entity_id: int
    entity_name: str
    matched_alias: str | None = None


class ContextChunk(BaseModel):
    point_id: str
    score: float
    chunk_id: str
    document_id: str
    source_type: str
    entity_id: int | None = None
    entity_name: str | None = None
    content: str


class ContextAPIResponse(BaseModel):
    success: bool
    query: str
    detected_entities: list[DetectedEntity] = Field(default_factory=list)
    selected_context_count: int = 0
    retrieval_confidence: str = "low"  # "low" | "medium" | "high"
    confidence: str = "low"
    contexts: list[ContextChunk] = Field(default_factory=list)
    context_text: str = ""

    @property
    def has_any_context(self) -> bool:
        return bool(self.contexts) and bool(self.context_text.strip())


# Ordinal ranking so "is this confidence >= our threshold" is a simple comparison
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def confidence_meets_threshold(confidence: str, minimum: str) -> bool:
    """
    e.g. confidence_meets_threshold("high", "medium") -> True
         confidence_meets_threshold("low", "medium")  -> False

    Unknown confidence strings are treated as "low" (fail safe: when in
    doubt, don't trust the retrieval enough to answer from it).
    """
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(minimum, 1)