"""The google-genai wrapper, against stand-ins shaped like genai.Client (no network, no google-genai)."""

import asyncio
from enum import Enum
from types import SimpleNamespace as NS

import pytest

import inference_tracker
from inference_tracker import context
from inference_tracker.google import track


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    return out


class Modality(Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"


class ClientError(Exception):
    def __init__(self):
        super().__init__("the message quotes the prompt")
        self.status = "RESOURCE_EXHAUSTED"


IMAGE_RESPONSE = NS(
    response_id="vertex-resp-1",
    usage_metadata=NS(
        prompt_token_count=1300,
        candidates_token_count=1150,
        thoughts_token_count=None,
        cached_content_token_count=None,
        prompt_tokens_details=[
            NS(modality=Modality.TEXT, token_count=40),
            NS(modality=Modality.IMAGE, token_count=1260),
        ],
        candidates_tokens_details=[
            NS(modality=Modality.IMAGE, token_count=1120),
            NS(modality=Modality.TEXT, token_count=30),
        ],
    ),
)


class Models:
    def __init__(self, fail=False):
        self.fail = fail

    def generate_content(self, *, model, contents, config=None):
        if self.fail:
            raise ClientError()
        return IMAGE_RESPONSE

    def generate_videos(self, *, model, prompt=None, image=None, config=None):
        return NS(name="projects/p/locations/us-central1/operations/op-1", done=False)

    def count_tokens(self, **kwargs):
        return NS(total_tokens=5)


class Operations:
    def __init__(self, final):
        self.final = final
        self.polls = 0

    def get(self, operation):
        self.polls += 1
        return self.final if self.polls >= 2 else NS(name=operation.name, done=False)


class AsyncModels:
    async def generate_content(self, *, model, contents, config=None):
        return IMAGE_RESPONSE


def client(final=None, fail=False):
    c = NS(
        _api_client=NS(vertexai=True, project="acme-prod", location="global"),
        models=Models(fail),
        operations=Operations(final),
        aio=NS(models=AsyncModels()),
    )
    return c


def test_generate_content_with_tokens_by_kind(records):
    config = {"image_config": {"aspect_ratio": "9:16", "image_size": "2K"}, "response_modalities": ["IMAGE"]}
    with context(user_id="u1", task_id="t1"):
        out = track(client()).models.generate_content(
            model="gemini-3.1-flash-image", contents=["a photo of my kid"], config=config
        )
    assert out is IMAGE_RESPONSE
    (r,) = records
    assert (r.provider, r.phase, r.status, r.request_id, r.model) == (
        "gcp.vertex_ai",
        "call",
        "ok",
        "vertex-resp-1",
        "gemini-3.1-flash-image",
    )
    assert r.units == {
        "input_tokens": 1300,
        "output_tokens": 1150,
        "input_text_tokens": 40,
        "input_image_tokens": 1260,
        "output_image_tokens": 1120,
        "output_text_tokens": 30,
    }
    assert r.settings == {"aspect_ratio": "9:16", "image_size": "2K", "gcp_project": "acme-prod"}
    assert (r.user_id, r.task_id) == ("u1", "t1")
    assert "kid" not in repr(r.to_wire())


def test_error_status_is_recorded_and_raised(records):
    with pytest.raises(ClientError):
        track(client(fail=True)).models.generate_content(model="gemini-3.1-flash-image", contents=["x"])
    (r,) = records
    assert (r.status, r.error_type) == ("error", "RESOURCE_EXHAUSTED")


def test_veo_is_a_start_then_a_finish_when_the_operation_is_done(records):
    final = NS(
        name="projects/p/locations/us-central1/operations/op-1",
        done=True,
        error=None,
        response=NS(generated_videos=[NS(video=NS(uri="gs://..."))], rai_media_filtered_count=0),
    )
    c = track(client(final))
    config = NS(duration_seconds=8, resolution="720p", generate_audio=True, number_of_videos=1, aspect_ratio="9:16")
    with context(user_id="u1", task_id="t1"):
        op = c.models.generate_videos(model="veo-3.1-fast-generate-001", prompt="...", config=config)
    while not op.done:
        op = c.operations.get(op)
    start, finish = records
    assert (start.phase, start.request_id) == ("start", "projects/p/locations/us-central1/operations/op-1")
    assert start.settings == {
        "duration_seconds": 8,
        "resolution": "720p",
        "generate_audio": True,
        "number_of_videos": 1,
        "aspect_ratio": "9:16",
        "gcp_project": "acme-prod",
    }
    assert (finish.phase, finish.status, finish.request_id) == ("finish", "ok", start.request_id)
    assert finish.units == {"videos": 1, "video_seconds": 8}
    assert (finish.user_id, finish.task_id) == ("u1", "t1")


def test_veo_filtered_is_an_error_finish(records):
    final = NS(
        name="projects/p/locations/us-central1/operations/op-1",
        done=True,
        error=None,
        response=NS(generated_videos=[], rai_media_filtered_count=1),
    )
    c = track(client(final))
    op = c.models.generate_videos(model="veo-3.1-fast-generate-001", prompt="...", config={"duration_seconds": 8})
    while not op.done:
        op = c.operations.get(op)
    assert (records[1].status, records[1].error_type, records[1].units) == (
        "error",
        "filtered",
        {"videos": 0, "video_seconds": 0},
    )


def test_async_and_pass_through(records):
    c = track(client())
    assert asyncio.run(c.aio.models.generate_content(model="gemini-3.1-flash-image", contents=["x"])) is IMAGE_RESPONSE
    assert c.models.count_tokens(model="m", contents=["x"]).total_tokens == 5
    assert len(records) == 1


def test_gemini_api_has_no_project(records):
    c = client()
    c._api_client = NS(vertexai=False, project=None)
    track(c).models.generate_content(model="gemini-3.1-flash", contents=["x"])
    assert records[0].provider == "gcp.gemini" and "gcp_project" not in records[0].settings


def test_a_call_the_app_stops_waiting_for_is_cancelled(records):
    class Slow:
        async def generate_content(self, **kwargs):
            await asyncio.sleep(60)

    c = client()
    c.aio = NS(models=Slow())

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(track(c).aio.models.generate_content(model="gemini-3.1-flash", contents=["x"]), 0.1)

    asyncio.run(main())
    (r,) = records
    assert (r.phase, r.status, r.error_type) == ("call", "cancelled", "CancelledError")


def test_without_a_client_nothing_is_done(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: False)
    track(client()).models.generate_content(model="gemini-3.1-flash-image", contents=["x"])
    assert out == []
