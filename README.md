# inferledger

Per-user inference cost for AI apps.

A small SDK records each inference call next to where it is made: who it was for, which
provider and model, the provider's request id, and the units and time it used. The provider's
bill decides how much was spent; the record decides who it was for. No prompts, images or
other user content are recorded.

Early work in progress.

## Setup

```python
import inferledger

inferledger.init(url="https://.../ingest", key="...", app="myapp")  # once per process
```

Or set `INFERLEDGER_URL`, `INFERLEDGER_KEY` and `INFERLEDGER_APP`. Without them nothing is sent
and every call below is a no-op.

Say who the work is for where it starts. Every tracked call inside the block is tagged:

```python
with inferledger.context(user_id=uid, task_id=task.id):
    ...
```

## fal

Wrap the client once; call it as before.

```python
import fal_client
import inferledger

client = inferledger.fal.track(fal_client.AsyncClient(key=FAL_KEY))

handle = await client.submit("fal-ai/kling-video/v2.6/pro/motion-control", arguments=args)
result = await handle.get()
```

`submit`, `subscribe`, `run` and the handle's `get` and `cancel` are recorded. Everything else
(`upload`, `status`, `stream`, ...) passes through. Tracking never raises and never changes the call.

When the result goes to a webhook, record the finish in the webhook handler:

```python
inferledger.fal.webhook(request_json)  # the body fal posts: request_id, status, payload
```

Your own fal apps make calls of their own. Name them, and the user, task and parent call travel
inside the request:

```python
client = inferledger.fal.track(fal_client.SyncClient(key=FAL_KEY), carry_to=["acme/"])
```

In the app, restore the context from the input before its own tracked calls:

```python
with inferledger.restore(input):
    ...
```

The `inferledger` field has to be accepted by the app's input model (an optional `dict` field,
or `extra="allow"`), or pydantic drops it before the handler sees it.

## What a record holds

One record per call, or two when the result arrives somewhere else (a "start" when it is sent, a
"finish" when the result comes back; the server joins them on provider + request id). The fields:
provider, model, the provider's request id, status and error type (an error's class name or the
provider's code, never a message), start time and duration, usage units as the provider reports
them, the settings that change the price (copied by name, never the whole input), and from the
context: user, task, and the record id of the parent call.
