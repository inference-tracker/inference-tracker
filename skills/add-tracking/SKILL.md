---
name: add-tracking
description: Add inference-tracker to one app, through a pull request, so every AI call it makes is recorded with the user and task it was for, and its exact cost can be taken from the provider's bill. Use when asked to track an app's AI calls, add per-user inference cost, or install inference-tracker in a repo. Starts from the list the find-ai-calls guide writes.
---

# Adding inference-tracker to an app

**The goal:** for every AI call the app makes, know who it was for (user, task) and what it cost. The
app records the "who" at the moment of the call. The cost comes later from the provider's bill, never
from a price list. Prompts, images and outputs are never recorded.

Read the library's README first, and each provider's current docs for anything below that touches
them. Provider APIs change; this guide only tells you what to look for.

## How to work

- **One repo at a time, through a pull request.** Work on a new branch; never push to the default
  branch or deploy to production yourself. The PR says what is tracked, what isn't yet, and why.
- **Stop and ask** where only a person can decide, instead of guessing. Mainly: who a call is for when
  the code doesn't say; whether code that looks dead still runs; which environment is test; whether a
  library change is wanted. Ask once, with the options and what each would mean.
- **The calls themselves don't change.** Same arguments, same results, same errors. If a change to a
  call seems needed, ask.

## 1. Start from the list of calls

Run the find-ai-calls guide on the repo first (or read the list it wrote). Check it is still current:
same commit, or nothing that touches AI calls changed since. Everything below works through that list,
line by line; a call that isn't on it is added to it first.

For each call, the list says how the result comes back:

- **direct**: the call returns the result.
- **queue + webhook**: the call is submitted, and the result arrives later somewhere else.
- **polled**: the call returns an operation or job id that is checked until done (Veo, a GPU host's
  status route).
- **stream**: the answer arrives in pieces; the usage comes with the last one.
- **written to the database**: the worker writes the result onto the task itself, with no webhook. The
  finish is recorded where the app notices the task changed, or by the worker.
- **inside another call**: the call is made by one of the app's own serverless apps, or by a tool
  inside an agent run, which was itself called by the app.

Each needs its start and its finish recorded, wherever they happen.

## 2. Decide who each call is for

Find where the app knows the user and the task: usually where a task is created. Wrap that work:

```python
with inference_tracker.context(user_id=user_id, task_id=task_id):
    ...  # every tracked call in here gets this user and task
```

Use the app's own stable user id (the one its analytics and payments use). Never an email.

When the call happens somewhere that doesn't know the user, don't send the user id there. Link the
call to its **parent** instead, and the dashboard finds the user by walking up the parents:

- **A webhook** gets the result of a call another process sent. Record the finish with the
  provider's webhook helper (`inference_tracker.fal.webhook(body)`); it joins the start by request id.
- **The app's own fal apps**: wrap the endpoint in `inference_tracker.fal.inside(self)`. Every call
  made inside gets the fal request the app is serving as its parent.
- **fal workflows**: submit with the header `inference_tracker.fal.WORKFLOW_ENVELOPE_HEADER: "true"`
  (only to `workflows/...` addresses). The result then lists each step, and each step is recorded
  with the run as its parent. The app must read `output` from the result.

## 3. Wrap the clients

Wrap each client where it is created. The calls themselves don't change, and if recording fails the
call still runs.

| Package | Wrap with | Tracked |
|---|---|---|
| `fal_client` | `inference_tracker.fal.track(client)` | submit, subscribe, run, results, webhooks |
| `openai` (OpenAI and Azure) | `inference_tracker.openai.track(client)` | responses, chat completions, images, raw `post` to an images route |
| `google-genai` | `inference_tracker.google.track(client)` | generate_content, images, Veo (start and finish) |

A call the wrappers don't cover (another SDK, raw HTTP, a layer like LiteLLM): add a wrapper to the
library, with tests, in its own pull request, rather than recording it by hand in the app. Record by
hand (`inference_tracker.record(...)`) only as a stopgap, and say so in the PR.

Check what each wrapper records against the call's real response: run it once against the real
SDK with a fake network, and look at the record. Usage the bill is split by (tokens by kind, seconds
of video, images) must be there.

## 4. Start the tracker

Once per process, at startup:

```python
inference_tracker.init(url=..., key=..., app="MyApp", environment="prod" or "test")
```

- The key comes from a secret, one per app. Without a key nothing is sent and the app runs as before.
- **Serverless functions that freeze between requests** (Cloud Functions, Lambda): pass
  `background=False`, and `flush=True` on the context, so records are sent before the function ends.
- Local emulators often run with dummy secrets: send nothing there.

## 5. Deploy to test, and look at the records

Make a few real generations in test, then check:

- Every call has a user, or a parent that leads to one. A call whose parent was never recorded
  means something else called it (another app, a playground, a test tool): find out what.
- Every `start` has its `finish`, except calls still running.
- Batches report 0 records lost.
- No record holds a prompt, an image, a URL or an output.

Then open the PR for review. Production comes after it's merged, by whoever deploys the app.

## 6. Connect the cost

The dashboard needs the provider's bill. Each kind of bill gives cost in its own way:

- **Per request id** (fal's models): exact, matched by the request id on the record.
- **A total per hour or day per model and kind** (Vertex per hour, Azure per day): each line is
  split over the recorded calls by their usage of that kind: tokens in, tokens out, image tokens,
  seconds of video.
- **Machine time** (your own serverless apps on fal, GPU hosts): the machine's bill for the hour is
  split over the requests that ran on it, evenly when several ran at once. What's left is idle time,
  shown per app.

## 7. Check the numbers

Take whole hours or a whole day, before any credits or discounts the provider adds later. Two checks:

1. **The bill is copied right.** The provider's own dashboard and the copy of its bill must match
   exactly, line by line. Watch the time range: compare exactly the same hours, in UTC.
2. **The records explain the bill.** Of the bill, how much is tied to a user, and what is the rest?
   Every gap gets a name: machines that were on with nothing running, calls with no user (and why:
   their parent was never recorded, something else called them), calls that were never recorded.
   What's left unexplained must be about 0. If it isn't, something is calling the provider that
   the list of calls missed: add it to the list (and, if the find guide could have caught it, to that
   guide).

## Things that went wrong before

- A provider's date range can include the whole hour its end names. Compare the same hours.
- Providers can write the same name in different cases (an account in lowercase in one place,
  uppercase in another). Compare names ignoring case.
- Tell a fal workflow by its `workflows/` address, not by how the app labels it.
- Text-only calls (prompt moderation, prompt rewriting, text-to-video) can't be tied to a task by
  matching files. Only recording at the call finds their user.
- A test tool pointed at production apps puts its cost on production. Point test tools at test apps.
- A field that is stored never changes meaning. A new need gets a new field.
- An hourly bill puts a call in the hour it *finished*, not the hour it started. A video started at
  10:59 and done at 11:00 is on the 11:00 line. Match calls to bill hours by finish time.
- A call that fails the app's own check of the answer (a structured output that doesn't fit its
  format) was still answered, so it's still billed. Record its usage anyway. Whole calls missing
  show up as the same small share missing on every kind of usage (input, output, cached).
