"""Read IdeaL cards using its public batch search contract."""

from __future__ import annotations

import httpx


async def search_cards(
    client: httpx.AsyncClient,
    query: str,
    namespace: str,
    api_key: str = "",
) -> tuple[tuple[str, str], ...]:
    if not query.strip():
        return ()

    response = await client.post(
        "/v1/search/batch",
        json={
            "queries": [query],
            "top_k": 3,
            "entity_type": "memory_card",
            "channel": "latest",
            "search_type": "bm25",
            "namespace": namespace,
        },
        headers={"X-API-Key": api_key} if api_key else None,
    )
    response.raise_for_status()
    (hits,) = response.json()["results"]
    cards: dict[str, str] = {}
    for hit in hits:
        entity_id = hit.get("entity_id")
        content = hit.get("content") or {}
        if not entity_id or entity_id in cards or content.get("status") in {"archived", "retired"}:
            continue
        text = content.get("description") or content.get("text") or hit.get("snippet")
        if not isinstance(text, str) or not text.strip():
            continue
        name = hit.get("name") or entity_id
        cards[entity_id] = f"{name}: {text.strip()[:2000]}"
    return tuple(cards.items())
