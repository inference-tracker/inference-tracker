# inference-tracker

Per-user inference cost for AI apps.

A small Python library that records each inference call (each call to an AI model) where the app
makes it: who it was for, which provider and model, the provider's request id, and how long it
took. The provider's bill says how much was spent; the record says who it was for. Prompts, images
and other user content are never recorded.

Early work in progress. Python 3.10 or newer, no other packages needed.

## Setup

```python
import inference_tracker

inference_tracker.init(url="https://.../ingest", key="...", app="myapp")  # once per process
```

Or set `INFERENCE_TRACKER_URL`, `INFERENCE_TRACKER_KEY` and `INFERENCE_TRACKER_APP` (and optionally
`INFERENCE_TRACKER_ENVIRONMENT`, e.g. `prod`). Without url, key and app, nothing is sent and every call
below does nothing.

Records are sent in the background every few seconds. Some hosts freeze the process as soon as the
response is sent (Cloud Functions, Lambda, Cloud Run by default). There, use
`init(..., background=False)` and `context(..., flush=True)`, which sends the records when the
block ends.

Say who the work is for where it starts. Every tracked call inside the block gets that user and task:

```python
with inference_tracker.context(user_id=uid, task_id=task.id):
    ...
```

A call made by another call names that call as its parent, by provider and the provider's request
id, the way the bill knows it. The server finds the user and task by walking up the parents, so
code that only knows its parent doesn't need them:

```python
with inference_tracker.context(parent_provider="fal", parent_request_id=rid):
    ...
```

## fal

Wrap the client once, then call it as before.

```python
import fal_client
import inference_tracker

client = inference_tracker.fal.track(fal_client.AsyncClient(key=FAL_KEY))

handle = await client.submit("fal-ai/kling-video/v2.6/pro/motion-control", arguments=args)
result = await handle.get()
```

Recorded: `submit`, `subscribe`, `run`, and getting or cancelling a result (`handle.get()`,
`handle.cancel()`, `client.result(...)`, `client.cancel(...)`). Everything else (`upload`,
`status`, `stream`...) passes through untouched. Tracking never raises an error and never changes
the call.

When fal sends the result to a webhook, record the finish in the webhook handler:

```python
inference_tracker.fal.webhook(request_json)  # the body fal posts: request_id, status, payload
```

### Workflows

Submit a fal workflow with the envelope header, and its result comes back as `{output, nodes}`,
with the request id of each step. Each step is then recorded too, with the run as its parent:
from the webhook body, or from the result when it is fetched here (`get()`, `result()`,
`subscribe()`). The app reads its output from `output`.

```python
handle = await client.submit(
    "workflows/ACME-COMPUTE/prod-myapp-video",
    arguments=args,
    webhook_url=hook_url,
    headers={inference_tracker.fal.WORKFLOW_ENVELOPE_HEADER: "true"},
)
...
inference_tracker.fal.webhook(request_json)  # the run's finish, and one record per step
```

When a run fails, the step that failed is recorded with fal's error code. A step that fell back to
another model also gets a record for the model that failed.

### Our own fal apps

Our own fal apps make calls of their own. In the endpoint, `inside(self)` makes the request the app
is serving the parent of every tracked call in the block:

```python
@fal.endpoint("/")
async def generate(self, input: Input) -> Output:
    with inference_tracker.fal.inside(self):
        return await self._inference(input)
```

The request id comes from fal's `self.current_request`. Without one (the app runs outside fal), the
block does nothing.

## OpenAI and Azure (the `openai` package)

```python
from openai import OpenAI
import inference_tracker

client = inference_tracker.openai.track(OpenAI(api_key=...))
response = client.responses.parse(model="gpt-5.4-mini", input=..., text_format=...)
```

Works the same for `AsyncOpenAI`, `AzureOpenAI`, and an `OpenAI` client pointed at Azure AI Foundry
(`base_url="https://<resource>.services.ai.azure.com/openai/v1/"`). Tracked: `responses.create` /
`parse`, `chat.completions.create` / `parse`, `images.generate` / `edit`. Each is one record with the
response id, the model the response names, and the tokens by kind (input, cached input, output,
reasoning) or images. Azure calls also note their resource name, which is how Azure's bill is split.
Streamed calls pass through untracked.

## Google (the `google-genai` package)

```python
from google import genai
import inference_tracker

client = inference_tracker.google.track(genai.Client(vertexai=True, project=..., location=...))
response = client.models.generate_content(model="gemini-3.1-flash-image", contents=..., config=...)
```

Tracked, sync and `client.aio`: `models.generate_content`, `generate_images`, `edit_image` (one record
each, with tokens by kind: text and image, in and out, thinking, cached), and Veo's
`models.generate_videos`: a "start" when it is sent and a "finish" when `operations.get()` returns the
operation done, with the seconds of video made. Settings that change the price (resolution, length,
audio, image size, aspect ratio, thinking) and the Google Cloud project are noted.

## What a record holds

One record per call. A call whose result arrives somewhere else (a queued call, a webhook) gets two:
a "start" when it is sent and a "finish" when the result comes back, both with the provider's
request id.

Each record has: provider, model, the provider's request id, status and error type (the error's
class name or the provider's code, never its message), start time and duration, usage as the
provider reports it, and the settings that change the price (copied by name, never the whole
input). From the context: user, task, and the call that made this one (`parent_provider` and
`parent_request_id`, the provider's request id).

## If sending fails

A busy or unreachable server is retried with growing waits. Records that still can't be sent, or
that the server refuses, are written to the `inference_tracker` log as one JSON line each. Every batch
also tells the server how many records were lost since the last one that got through.
