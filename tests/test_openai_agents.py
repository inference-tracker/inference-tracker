"""A tracked AsyncOpenAI given to the OpenAI Agents SDK (openai-agents), over a fake network: the SDK accepts
it, and every model call of a run is recorded, plain or streamed, on OpenAI, Azure AI Foundry or Vertex AI.
Skipped when openai-agents isn't installed."""

import asyncio
import json

import pytest

agents = pytest.importorskip("agents")
openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from agents import (  # noqa: E402
    Agent,
    Model,
    ModelProvider,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

import inference_tracker  # noqa: E402
from inference_tracker import context  # noqa: E402
from inference_tracker.openai import track  # noqa: E402

VERTEX = "https://aiplatform.googleapis.com/v1beta1/projects/acme-prod/locations/global/endpoints/openapi"
AZURE = "https://acme-resource.services.ai.azure.com/openai/v1/"


@pytest.fixture(autouse=True)
def no_tracing():
    set_tracing_disabled(True)
    yield
    set_default_openai_api("responses")


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    return out


@function_tool
def lookup(city: str) -> str:
    """Looks up the weather in a city."""
    return "sunny"


class Server:
    """Answers each model call in turn: first with a call to the lookup tool, then with a message. As JSON,
    or as server-sent events when the request asks for a stream."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        body = json.loads(request.content)
        n = len(self.requests)
        chat = request.url.path.endswith("/chat/completions")
        answer = (chat_answer if chat else responses_answer)(n, body)
        if not body.get("stream"):
            return httpx.Response(200, json=answer, headers={"x-request-id": f"req_{n}"})
        events = (chat_chunks if chat else responses_events)(answer, body)
        sse = "".join(f"data: {json.dumps(e) if isinstance(e, dict) else e}\n\n" for e in events)
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


def responses_answer(n, body):
    if n == 1:
        output = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"city": "Paris"}',
                "status": "completed",
            }
        ]
    else:
        output = [
            {
                "type": "message",
                "id": f"msg_{n}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "It is sunny.", "annotations": []}],
            }
        ]
    return {
        "id": f"resp_{n}",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-mini-2026-03-17",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": {
            "input_tokens": 100 * n,
            "output_tokens": 10 * n,
            "total_tokens": 110 * n,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def responses_events(answer, body):
    started = {**answer, "status": "in_progress", "output": [], "usage": None}
    return [
        {"type": "response.created", "sequence_number": 0, "response": started},
        *(
            {"type": "response.output_item.done", "sequence_number": 1 + i, "output_index": i, "item": item}
            for i, item in enumerate(answer["output"])
        ),
        {"type": "response.completed", "sequence_number": 9, "response": answer},
    ]


def chat_answer(n, body):
    if n == 1:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"city": "Paris"}'}}
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": "It is sunny."}
        finish = "stop"
    return {
        "id": f"vertex-resp-{n}",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 100 * n, "completion_tokens": 10 * n, "total_tokens": 150 * n},
    }


def chat_chunks(answer, body):
    message = answer["choices"][0]["message"]
    delta = {"role": "assistant"}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": 0, **message["tool_calls"][0]}]
    else:
        delta["content"] = message["content"]
    base = {"id": answer["id"], "object": "chat.completion.chunk", "created": 0, "model": answer["model"]}
    finish = answer["choices"][0]["finish_reason"]
    out = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}]
    if (body.get("stream_options") or {}).get("include_usage"):
        out.append({**base, "choices": [], "usage": answer["usage"]})
    return out + ["[DONE]"]


def client(server, base_url="https://api.openai.com/v1/"):
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return track(openai.AsyncOpenAI(api_key="sk-test", base_url=base_url, http_client=http, max_retries=0))


def agent(model=None, settings=None):
    return Agent(
        name="weather",
        instructions="Answer about the weather.",
        tools=[lookup],
        model=model,
        model_settings=settings or ModelSettings(),
    )


def test_default_client_responses_run_sync_records_every_model_call(records):
    server = Server()
    set_default_openai_client(client(server), use_for_tracing=False)
    with context(user_id="u1", task_id="t1"):
        result = Runner.run_sync(agent("gpt-5.4-mini"), "What is the weather in a photo of my kid's city?")
    assert result.final_output == "It is sunny."
    assert [r.request_id for r in records] == ["resp_1", "resp_2"]
    assert all((r.provider, r.status, r.user_id, r.task_id) == ("openai", "ok", "u1", "t1") for r in records)
    assert [r.units["input_tokens"] for r in records] == [100, 200]
    assert "kid" not in repr([r.to_wire() for r in records])


def test_default_client_chat_completions_api(records):
    server = Server()
    set_default_openai_client(client(server), use_for_tracing=False)
    set_default_openai_api("chat_completions")
    result = asyncio.run(Runner.run(agent("gpt-5.4-mini"), "weather?"))
    assert result.final_output == "It is sunny."
    assert [r.request_id for r in records] == ["vertex-resp-1", "vertex-resp-2"]


def test_chat_completions_model_on_vertex(records):
    server = Server()
    model = OpenAIChatCompletionsModel(model="google/gemini-3.1-flash", openai_client=client(server, VERTEX))
    result = asyncio.run(Runner.run(agent(model), "weather?"))
    assert result.final_output == "It is sunny."
    first, second = records
    assert (first.provider, first.model, first.request_id) == ("gcp.vertex_ai", "gemini-3.1-flash", "vertex-resp-1")
    assert first.settings == {"gcp_project": "acme-prod", "gcp_location": "global"}
    assert first.units == {
        "input_tokens": 100,
        "output_tokens": 10,
        "thinking_tokens": 40,
        "input_text_tokens": 100,
        "output_text_tokens": 10,
    }
    assert second.units["thinking_tokens"] == 80


class Foundry(ModelProvider):
    """An app's own provider: every agent runs on Azure AI Foundry through the Responses API."""

    def __init__(self, openai_client):
        self.openai_client = openai_client

    def get_model(self, model_name: str | None) -> Model:
        return OpenAIResponsesModel(model=model_name or "gpt-5.4-mini", openai_client=self.openai_client)


def test_custom_model_provider_on_azure_foundry(records):
    server = Server()
    config = RunConfig(model_provider=Foundry(client(server, AZURE)))
    result = Runner.run_sync(agent(), "weather?", run_config=config)
    assert result.final_output == "It is sunny."
    assert [(r.provider, r.settings.get("azure_resource")) for r in records] == [
        ("azure.ai.openai", "acme-resource"),
        ("azure.ai.openai", "acme-resource"),
    ]


def run_streamed(starting_agent, **kwargs):
    async def run():
        result = Runner.run_streamed(starting_agent, "weather?", **kwargs)
        async for _ in result.stream_events():
            pass
        return result.final_output

    return asyncio.run(run())


def test_run_streamed_on_the_responses_api(records):
    server = Server()
    set_default_openai_client(client(server), use_for_tracing=False)
    assert run_streamed(agent("gpt-5.4-mini")) == "It is sunny."
    assert [(r.request_id, r.status, r.units["input_tokens"]) for r in records] == [
        ("resp_1", "ok", 100),
        ("resp_2", "ok", 200),
    ]


def test_run_streamed_on_vertex_chat_completions_with_usage_asked_for(records):
    server = Server()
    model = OpenAIChatCompletionsModel(model="google/gemini-3.1-flash", openai_client=client(server, VERTEX))
    assert run_streamed(agent(model, ModelSettings(include_usage=True))) == "It is sunny."
    assert [(r.provider, r.request_id, r.units.get("output_text_tokens")) for r in records] == [
        ("gcp.vertex_ai", "vertex-resp-1", 10),
        ("gcp.vertex_ai", "vertex-resp-2", 20),
    ]


def test_run_streamed_on_vertex_without_usage_asked_for_is_recorded_without_usage(records):
    """The SDK asks for usage in a chat completions stream only on api.openai.com, unless the agent's
    ModelSettings(include_usage=True) says so. The tracker doesn't ask for it (see openai.py)."""
    server = Server()
    model = OpenAIChatCompletionsModel(model="google/gemini-3.1-flash", openai_client=client(server, VERTEX))
    assert run_streamed(agent(model)) == "It is sunny."
    assert all("stream_options" not in json.loads(r.content) for r in server.requests)
    assert [(r.request_id, r.status, r.units) for r in records] == [
        ("vertex-resp-1", "ok", {}),
        ("vertex-resp-2", "ok", {}),
    ]


def test_runner_managed_retries_use_a_copy_of_the_client_that_is_tracked_too(records):
    """With the runner's own retry settings, the SDK calls client.with_options(max_retries=0)."""
    from agents import ModelRetrySettings

    server = Server()
    set_default_openai_client(client(server), use_for_tracing=False)
    settings = ModelSettings(retry=ModelRetrySettings(max_retries=0))
    assert Runner.run_sync(agent("gpt-5.4-mini", settings), "weather?").final_output == "It is sunny."
    assert [r.request_id for r in records] == ["resp_1", "resp_2"]
