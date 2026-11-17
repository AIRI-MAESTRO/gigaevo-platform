import json
from datetime import datetime, timezone

import httpx
import pytest

from common.ideal_search import search_cards
from common.llm_rate_limit import RateLimitedAsyncClient


@pytest.mark.asyncio
async def test_ideal_search_reads_current_response_and_scopes_namespace():
    def reply(request):
        assert request.url.path == "/v1/search/batch"
        assert request.headers["X-API-Key"] == "test-key"
        assert json.loads(request.content) == {
            "queries": ["exact match"],
            "top_k": 3,
            "entity_type": "memory_card",
            "channel": "latest",
            "search_type": "bm25",
            "namespace": "smoke",
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    [
                        {"entity_id": "card-1", "name": "Color", "content": {"description": "Copy the input."}},
                        {"entity_id": "card-1", "name": "Duplicate", "content": {"description": "Ignore."}},
                    ]
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply), base_url="http://ideal") as client:
        assert await search_cards(client, "exact match", "smoke", "test-key") == (("card-1", "Color: Copy the input."),)


@pytest.mark.asyncio
async def test_429_retries_same_request_after_retry_after(monkeypatch):
    from common import llm_rate_limit

    requests, delays = [], []

    async def sleep(seconds):
        delays.append(seconds)

    def reply(request):
        requests.append(request.content)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "8"})
        return httpx.Response(200)

    monkeypatch.setattr(llm_rate_limit.asyncio, "sleep", sleep)
    async with RateLimitedAsyncClient(transport=httpx.MockTransport(reply), rate_limit_retries=2) as client:
        response = await client.post("http://model/chat/completions", json={"model": "test"})
    assert response.status_code == 200
    assert requests[0] == requests[1]
    assert delays == [8]


@pytest.mark.asyncio
async def test_429_never_retries_before_provider_window_or_past_retry_budget(monkeypatch):
    from common import llm_rate_limit

    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm_rate_limit.asyncio, "sleep", sleep)
    count = 0

    def reply(request):
        nonlocal count
        count += 1
        return httpx.Response(429, headers={"Retry-After": "3"})

    async with RateLimitedAsyncClient(transport=httpx.MockTransport(reply), rate_limit_retries=1) as client:
        assert (await client.get("http://model")).status_code == 429
    assert count == 2
    assert len(delays) == 1 and delays[0] >= 3


def test_retry_after_accepts_http_date_and_bounds_provider_delay():
    from common.llm_rate_limit import retry_delay

    now = datetime(2026, 11, 17, tzinfo=timezone.utc)
    assert retry_delay(httpx.Headers({"Retry-After": "Tue, 17 Nov 2026 00:00:20 GMT"}), 0, now) == 20
    assert retry_delay(httpx.Headers({"Retry-After": "600"}), 0, now) is None


@pytest.mark.parametrize("executed", [False, True])
def test_chain_validity_requires_a_successful_training_execution(tmp_path, monkeypatch, executed):
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    template = Path(__file__).parents[1] / "master_api/src/folder_constructor/validate_templates/chain/validate.py"
    module_path = tmp_path / "validate.py"
    module_path.write_text(
        template.read_text()
        .replace("${fitness_mode}", "accuracy")
        .replace("${validation_type}", "Binary (0/1)")
        .replace("${metric}", "equality")
    )
    monkeypatch.setitem(
        sys.modules,
        "helper",
        SimpleNamespace(
            parse_chain_config_json=json.loads,
            execute_chain_on_dataset=lambda *_: [
                {"prediction": "", "ground_truth": "blue", "success": executed, "time": 0.0},
            ],
            _extract_numeric_answer=lambda _: None,
            _extract_prediction=lambda text: text,
        ),
    )
    monkeypatch.setitem(sys.modules, "feedback_integration", None)
    monkeypatch.setenv("PROBLEM_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    validate = runpy.run_path(str(module_path))["validate"]
    metrics, _ = validate(
        {"train_dataset": [1], "val_dataset": [], "target_column": "expected"},
        {"chain_config_json": "{}"},
    )
    assert metrics["is_valid"] == float(executed)
