# inferledger

Per-user inference cost for AI apps.

A small Python library that records each inference call (each call to an AI model) where the app
makes it: who it was for, which provider and model, the provider's request id, and how long it
took. The provider's bill says how much was spent; the record says who it was for. Prompts, images
and other user content are never recorded.

Early work in progress. Python 3.10 or newer, no other packages needed.

## Setup

```python
import inferledger

inferledger.init(url="https://.../ingest", key="...", app="myapp")  # once per process
```

Or set `INFERLEDGER_URL`, `INFERLEDGER_KEY` and `INFERLEDGER_APP` (and optionally
`INFERLEDGER_ENVIRONMENT`, e.g. `prod`). Without url, key and app, nothing is sent and every call
below does nothing.

Records are sent in the background every few seconds. Some hosts freeze the process as soon as the
response is sent (Cloud Functions, Lambda, Cloud Run by default). There, use
`init(..., background=False)` and `context(..., flush=True)`, which sends the records when the
block ends.

Say who the work is for where it starts. Every tracked call inside the block gets that user and task:

```python
with inferledger.context(user_id=uid, task_id=task.id):
    ...
```

## fal

Wrap the client once, then call it as before.

```python
import fal_client
import inferledger

client = inferledger.fal.track(fal_client.AsyncClient(key=FAL_KEY))

handle = await client.submit("fal-ai/kling-video/v2.6/pro/motion-control", arguments=args)
result = await handle.get()
```

Recorded: `submit`, `subscribe`, `run`, and getting or cancelling a result (`handle.get()`,
`handle.cancel()`, `client.result(...)`, `client.cancel(...)`). Everything else (`upload`,
`status`, `stream`...) passes through untouched. Tracking never raises an error and never changes
the call.

When fal sends the result to a webhook, record the finish in the webhook handler:

```python
inferledger.fal.webhook(request_json)  # the body fal posts: request_id, status, payload
```

### Our own fal apps

Our own fal apps make calls of their own. List their names (or the start of their names) in
`carry_to`. Calls to those apps then carry the user, the task and the calling record's id, inside
the request:

```python
client = inferledger.fal.track(fal_client.SyncClient(key=FAL_KEY), carry_to=["acme/"])
```

Never list fal's public models there: they may refuse an input they don't know.

Inside the app, restore that before its own tracked calls:

```python
with inferledger.restore(input):
    ...
```

The app's input model must accept the `inferledger` field (an optional `dict` field, or
`extra="allow"`). Otherwise pydantic drops it before the handler sees it.

## What a record holds

One record per call. A call whose result arrives somewhere else (a queued call, a webhook) gets two:
a "start" when it is sent and a "finish" when the result comes back, both with the provider's
request id.

Each record has: provider, model, the provider's request id, status and error type (the error's
class name or the provider's code, never its message), start time and duration, usage as the
provider reports it, and the settings that change the price (copied by name, never the whole
input). From the context: user, task, and `parent_id`, the record id of the call that made this one.

## If sending fails

A busy or unreachable server is retried with growing waits. Records that still can't be sent, or
that the server refuses, are written to the `inferledger` log as one JSON line each. Every batch
also tells the server how many records were lost since the last one that got through.
