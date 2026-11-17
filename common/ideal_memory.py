"""IdeaL read provider for the public GigaEvo Core memory-guided pipeline."""

from __future__ import annotations

import os
import re

import httpx
from gigaevo.memory.provider import MemoryProvider
from gigaevo.memory.read.reader import MemorySelection
from loguru import logger

from common.ideal_search import search_cards


class IdeaLMemoryProvider(MemoryProvider):
    async def select_cards(
        self,
        program,
        *,
        task_description,
        metrics_description,
        parent_context=None,
        pending_counts=None,
    ) -> MemorySelection:
        # The Platform prompt template contains many generic instructions.
        # Prefer the actual task description over the full evaluator prompt.
        description = re.search(r"^[- ]*Task description:\s*(.+)$", task_description, re.MULTILINE)
        query = os.getenv("MEMORY_QUERY") or (description[1] if description else task_description)
        namespace = os.environ["MEMORY_NAMESPACE"]
        try:
            async with httpx.AsyncClient(
                base_url=os.environ["MEMORY_API_URL"],
                timeout=10.0,
            ) as client:
                cards = await search_cards(
                    client,
                    query.strip()[:500],
                    namespace,
                    os.getenv("MEMORY_API_KEY", ""),
                )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
            logger.warning("IdeaL retrieval failed for namespace {} ({})", namespace, type(error).__name__)
            return MemorySelection()

        ids = tuple(card_id for card_id, _ in cards)
        logger.info("IdeaL selected {} card(s) for namespace {}: {}", len(ids), namespace, ", ".join(ids))
        return MemorySelection(cards=tuple(text for _, text in cards), card_ids=ids)
