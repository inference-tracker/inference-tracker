"""The google wrapper around the real google-genai and vertexai packages, with Google's servers stood in by
fakes: an httpx mock for google-genai, and the requests library's HTTP adapter for vertexai. Skipped when the
packages aren't installed."""

import asyncio
import json

import pytest

genai = pytest.importorskip("google.genai")
httpx = pytest.importorskip("httpx")

from google.genai import types  # noqa: E402
from google.oauth2.credentials import Credentials  # noqa: E402

import inference_tracker  # noqa: E402
from inference_tracker import context  # noqa: E402
from inference_tracker.google import track  # noqa: E402


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    return out


def google(request):
    """Vertex's predict (gemini-embedding-001, text-embedding-005), Vertex's embedContent (newer Gemini
    embedding models, one content at a time), and the Gemini API's batchEmbedContents."""
    path = request.url.path
    if path.endswith(":predict"):
        n = len(json.loads(request.content)["instances"])
        return httpx.Response(
            200,
            json={
                "predictions": [
                    {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 7, "truncated": False}}}
                ]
                * n,
                "metadata": {"billableCharacterCount": 40},
            },
        )
    if path.endswith(":embedContent"):
        return httpx.Response(
            200,
            json={
                "embedding": {"values": [0.1, 0.2]},
                "usageMetadata": {
                    "promptTokenCount": 300,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 10},
                        {"modality": "IMAGE", "tokenCount": 290},
                    ],
                },
            },
        )
    n = len(json.loads(request.content)["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}] * n})


def client(vertexai=True):
    transport = httpx.MockTransport(google)
    options = types.HttpOptions(
        httpx_client=httpx.Client(transport=transport), httpx_async_client=httpx.AsyncClient(transport=transport)
    )
    if not vertexai:
        return track(genai.Client(api_key="test-key", http_options=options))
    return track(
        genai.Client(
            vertexai=True,
            project="acme-prod",
            location="us-central1",
            credentials=Credentials(token="test-token"),
            http_options=options,
        )
    )


def test_vertex_embed_content_records_the_tokens_vertex_reports(records):
    with context(user_id="u1", task_id="t1"):
        out = client().models.embed_content(model="gemini-embedding-001", contents=["a photo of my kid", "hello"])
    assert len(out.embeddings) == 2
    (r,) = records
    assert (r.provider, r.model, r.status, r.request_id) == ("gcp.vertex_ai", "gemini-embedding-001", "ok", None)
    assert r.units == {
        "inputs": 2,
        "input_tokens": 14,
        "input_text_tokens": 14,
        "billable_characters": 40,
    }
    assert r.settings == {"gcp_project": "acme-prod"}
    assert (r.user_id, r.task_id) == ("u1", "t1")
    assert "kid" not in repr(r.to_wire())


def test_vertex_embed_content_async_with_an_image_records_tokens_by_modality(records):
    content = types.Content(parts=[types.Part(text="hi"), types.Part.from_bytes(data=b"...", mime_type="image/png")])
    asyncio.run(client().aio.models.embed_content(model="gemini-embedding-2", contents=content))
    (r,) = records
    assert r.units == {"inputs": 1, "input_tokens": 300, "input_text_tokens": 10, "input_image_tokens": 290}


def test_gemini_api_embed_content_has_no_token_count_so_characters_are_counted(records):
    client(vertexai=False).models.embed_content(model="gemini-embedding-001", contents=["abc", "de"])
    (r,) = records
    assert (r.provider, r.model) == ("gcp.gemini", "gemini-embedding-001")
    assert r.units == {"inputs": 2, "input_characters": 5}


# ---------- the older vertexai package ----------


@pytest.fixture
def vertexai(monkeypatch):
    """vertexai, set up with test credentials and REST, and every HTTP request it sends answered here."""
    vertexai = pytest.importorskip("vertexai")
    requests = pytest.importorskip("requests")
    from requests.adapters import HTTPAdapter

    sent = []

    def send(self, request, **kwargs):
        sent.append(request)
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.url = request.url
        response.headers["content-type"] = "application/json"
        if request.method == "GET":  # from_pretrained looks the model up in Model Garden
            name = request.path_url.split("?")[0].split("/v1/")[-1]
            body = {
                "name": name,
                "versionId": "005",
                "launchStage": "GA",
                "openSourceCategory": "PROPRIETARY",
                "predictSchemata": {
                    "instanceSchemaUri": "gs://google-cloud-aiplatform/schema/predict/instance/text_embedding_1.0.0.yaml"
                },
                "publisherModelTemplate": "projects/{project}/locations/{location}/" + name,
            }
        else:
            n = len(json.loads(request.body)["instances"])
            body = {
                "predictions": [
                    {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 6, "truncated": False}}}
                ]
                * n,
                "metadata": {"billableCharacterCount": 33},
            }
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(HTTPAdapter, "send", send)
    vertexai.init(
        project="acme-prod", location="us-central1", credentials=Credentials(token="test-token"), api_transport="rest"
    )
    return sent


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_text_embedding_model(records, vertexai):
    from vertexai.language_models import TextEmbeddingModel

    model = track(TextEmbeddingModel.from_pretrained("text-embedding-005"))
    with context(user_id="u1"):
        out = model.get_embeddings(["a photo of my kid", "hello", "world"], output_dimensionality=2)
    assert [len(e.values) for e in out] == [2, 2, 2]
    assert (
        vertexai[-1]
        .url.split("?")[0]
        .endswith("/projects/acme-prod/locations/us-central1/publishers/google/models/text-embedding-005:predict")
    )
    (r,) = records
    assert (r.provider, r.model, r.status, r.user_id) == ("gcp.vertex_ai", "text-embedding-005", "ok", "u1")
    assert r.units == {"inputs": 3, "input_tokens": 18, "input_text_tokens": 18, "billable_characters": 33}
    assert r.settings == {"gcp_project": "acme-prod", "gcp_location": "us-central1"}
    assert "kid" not in repr(r.to_wire())
    assert model.count_tokens is not None  # everything else is the model's own


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_text_embedding_model_async(records, vertexai, monkeypatch):
    """vertexai's async calls go over gRPC, or REST with async credentials; the fake network here is the
    requests library's. So the endpoint's predict_async is answered by its own predict, over that network."""
    from vertexai.language_models import TextEmbeddingModel

    raw = TextEmbeddingModel.from_pretrained("text-embedding-005")

    async def predict_async(**kwargs):
        return raw._endpoint.predict(**kwargs)

    monkeypatch.setattr(raw._endpoint, "predict_async", predict_async)
    out = asyncio.run(track(raw).get_embeddings_async(["hello"]))
    assert len(out) == 1
    (r,) = records
    assert r.units == {"inputs": 1, "input_tokens": 6, "input_text_tokens": 6, "billable_characters": 33}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_text_embedding_model_error_is_recorded_and_raised(records, vertexai):
    from vertexai.language_models import TextEmbeddingModel

    model = track(TextEmbeddingModel.from_pretrained("text-embedding-005"))
    with pytest.raises(TypeError):
        model.get_embeddings("not a list")
    (r,) = records
    assert (r.status, r.error_type) == ("error", "TypeError")
