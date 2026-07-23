"""
Thin async client around your existing context-retrieval API.

ASSUMPTION (please confirm/adjust): I'm assuming the API takes a POST
request with a JSON body `{"query": "<text>"}` at
`{base_url}{query_path}`, based on the response shape you shared. If
your real endpoint uses GET with a query param instead, only
`get_context()` below needs to change — nothing else in the app talks
to HTTP directly.
"""
import logging

import httpx

from app.clients.context_api_schemas import ContextAPIResponse
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ContextAPIError(Exception):
    """Raised when the context API is unreachable or returns a bad response."""


class ContextAPIClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        settings = get_settings()
        self._base_url = settings.context_api_base_url.rstrip("/")
        self._query_path = settings.context_api_retrieve_path
        self._timeout = settings.context_api_timeout_seconds
        # Allow injecting a client (e.g. one backed by httpx.MockTransport)
        # so tests never make real network calls.
        self._client = client or httpx.AsyncClient(timeout=self._timeout)

    async def get_context(self, query: str) -> ContextAPIResponse:
        """
        Calls the context API with a standalone (already-rewritten)
        query string and returns a typed response.

        Raises ContextAPIError on network failure or malformed response
        — callers should treat that as "could not retrieve" and fall
        back to a refusal, not let it bubble up as a 500.
        """
        url = f"{self._base_url}{self._query_path}"
        try:
            resp = await self._client.get(url, params={"query": query})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Context API call failed for query=%r: %s", query, exc)
            raise ContextAPIError(f"Context API request failed: {exc}") from exc

        try:
            data = resp.json()
            return ContextAPIResponse.model_validate(data)
        except (ValueError, TypeError) as exc:
            logger.error("Context API returned unparseable response: %s", exc)
            raise ContextAPIError(f"Context API returned invalid JSON: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()