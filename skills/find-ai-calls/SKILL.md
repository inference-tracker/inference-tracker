---
name: find-ai-calls
description: Find every AI call in the current repo or folder (provider, model, where in the code, how it's called, who it's for) and which ones inference-tracker doesn't record yet. Changes no code; saves the list to .inference-tracker/calls.md, which the add-tracking command works from. Use when asked to find, list or count AI calls, to see what isn't tracked, or before adding tracking to an app.
---

# Finding every AI call

**The goal:** a list of every place the code sends work to an AI model or to a machine that runs one,
saying for each whether inference-tracker already records it, with enough detail to track the rest:
who it's for, and which bill it lands on. The add-tracking command works from this list and doesn't
search again, so the list must be complete.

**This changes no code.** No edits, no commits, no branch switches, no installs, no deploys, and no
calls to the providers. Its only output is one file, `.inference-tracker/calls.md`, in the folder it
was run in (step 5), plus a short summary in the chat.

Code reading finds candidates. It can't see a provider chosen at runtime, calls made from outside the
code on the same keys, or tell for sure live code from dead. Say so in the report instead of guessing;
the bill settles it later (the add-tracking guide, "Check the numbers").

## 1. Read the code that's deployed

- **Find what actually runs.** The default branch (`git symbolic-ref refs/remotes/origin/HEAD`) is
  often it, but not always: read the CI and deploy files to see which branch or tag ships to
  production, and compare. Apps deployed one by one from tags may run older code than the branch.
- **Apps deployed one by one from tags:** take each app's newest production tag and compare its AI-call
  code with the default branch; say where they differ. An app with no deploy record (no tags, deployed
  by hand): say so.
- **Also read the branch the team merges into** (the next release; see recent merged pull requests).
  Calls that exist only there are listed as "coming": add-tracking works on that branch.
- **Read it as the remote has it.** Fetch (this changes no files), then read that branch or tag, for
  example by exporting it with `git archive` to a scratch folder. Never switch the checkout's branch.
  Note uncommitted work or a remote that no longer exists.
- Note each repo's commit and date in the report, so it's clear how fresh the list is.
- **Copies:** a folder that is a copy or a git worktree of another repo isn't counted twice: say what it
  copies and whether its AI calls differ. Apps made from the same template: do one fully, then diff the
  others against it and report each app's own lines, with a "Differences from ..." section.
- When given a whole folder, look at every repo: backends, serverless and GPU apps, mobile and web apps
  (they can call providers directly), workers, scripts, notebooks, CI, internal tools.
- Read the project's own notes (docs, backlog) for known gaps and deliberate choices. Trust the code
  over the notes where they disagree, and say so.

## 2. Search wide

**The rule: list every outside address the code sends requests to, and classify each one.** The lists
below help find them. They are starting points, not limits: a vendor not listed here (a speech, vision
or search API, a model marketplace) is still an AI call.

**Packages and methods** (in every language the repo uses):

- **fal:** `fal_client` (clients, and module-level `fal_client.submit` / `run` / `stream`), subclasses
  of its clients, request handles built by hand, `fal.App`, `@fal-ai/client`. Endpoint ids don't all
  start with `fal-ai/` (`workflows/...`, `xai/...`, `bytedance/...`). `api.fal.ai` is fal's admin API,
  not inference.
- **OpenAI protocol** (OpenAI, Azure, and anything that speaks it): `OpenAI(`, `AsyncOpenAI(`,
  `AzureOpenAI(`, the old `OpenAIApi` / `createChatCompletion` / `basePath`; methods `responses.*`,
  `chat.completions.*`, `beta.chat.completions.*`, `images.*`, `videos.*`, `embeddings.*`,
  `moderations.create`, `files` / `vector_stores`, a raw `client.post(`, `with_raw_response`; built-in
  tools inside a response (image generation, web search, file search), which are billed inside it.
- **Google:** `google.genai` (`generate_content`, `generate_images`, `generate_videos`,
  `operations.get`, `embed_content`), `google.generativeai`, `vertexai` (including
  `vertexai.language_models`), `@google/genai`; Cloud Vision, Speech, Translate.
- **Anthropic and AWS:** `anthropic`, `@anthropic-ai/sdk`, `boto3` with `bedrock-runtime`, and AWS's
  other AI services (Rekognition, Textract, Comprehend, Polly, Transcribe, Translate).
- **Layers over a provider:** the OpenAI Agents SDK (`Runner.run` / `run_sync` / `run_streamed`,
  `ModelProvider`, `OpenAIChatCompletionsModel`, `OpenAIResponsesModel`, `LitellmModel`,
  `set_default_openai_client`, `@openai/agents`), `litellm`, `langchain`, `llama_index`, `pydantic_ai`,
  `ai` and `@ai-sdk/*` (Vercel). The real provider is underneath: read the model string
  (`bedrock/...`, `vertex_ai/...`) or the client and address it's given.
- **Machines:** `import modal` / `modal.App` (searching `modal` alone finds UI dialogs), `runpod`,
  `replicate`, Hugging Face `InferenceClient`, Together, Groq, Mistral, Cohere, ElevenLabs, Deepgram,
  AssemblyAI, Stability.
- **AI inside other services:** BigQuery `ML.GENERATE_TEXT` / `AI.GENERATE`, Cloudflare Workers AI.

**Addresses** (raw HTTP is the easiest to miss: `requests`, `httpx`, `aiohttp`, `urllib.request`,
`fetch`, `axios`, and `wss://` streams):

- `fal.run`, `queue.fal.run`
- `api.openai.com`, `.openai.azure.com`, `.services.ai.azure.com`, `cognitiveservices.azure.com`
- `generativelanguage.googleapis.com`, `aiplatform.googleapis.com` (including its `/openapi`
  OpenAI-compatible path), `vision.googleapis.com`, `speech.googleapis.com`
- `api.anthropic.com`, `bedrock-runtime.`, `api.runpod.ai`, `.modal.run`, `api.replicate.com`
- any address read from config or an environment variable (`*_URL`, `*_ENDPOINT`) that ends up in a
  request. Auth headers give them away too: `Authorization: Bearer`, `Authorization: Key`, `x-api-key`.
- A model can be named only in a header (Azure's image deployment header), and a provider key can be
  passed inside a payload ("bring your own key"): then one call is on two bills.

**Keys:** environment variables and secrets (`*_API_KEY`, `*_TOKEN`, `BEARER_TOKEN`, `FAL_*`,
`OPENAI_*`, `AZURE_*`, `FOUNDRY_*`, `RUNPOD_*`, `MODAL_*`, service-account files), and each function's
own list of secrets (`SecretParam(`, `secrets=[...]`): a function can only call providers whose keys
it gets, which narrows a runtime choice. Some apps load every secret from a secret store at startup:
then the list can't be complete from code; say so. Check defaults: an environment setting that defaults
to production means a test worker uses production keys.

**What runs where:** deploy lists and entry points (a Firebase `main.py`'s exports, including ones only
exported in production; a handler that isn't exported isn't deployed), Cloud Run and Dockerfiles, fal
apps in `pyproject.toml` (`[tool.fal.apps]`) and their machine settings in CI (`fal apps scale`), Modal
and RunPod deploy scripts. Switches that look live but aren't: a percentage or random split set to
always one side, a flag that's off.

**Where a runtime choice is read:** database collections and config files that name the provider,
model, endpoint or key per task. With the user's OK, read the production config itself, read-only: it
settles most "can't tell" items. Seed files and one-off admin scripts that update them hint at the
current values (docs can be out of date).

**From the mobile or web app:** direct calls to providers, the backend functions it calls
(`httpsCallable`), and database documents it writes that start backend work (a task with a status).
On-device AI (models run on the phone) isn't billed: note it separately.

**Not inference, but note it:** SDKs that send traces with prompts to a provider (the Agents SDK's
trace export), AI in CI (code review actions, `secrets: inherit`), dev scripts that use production keys
(translation helpers in a UI repo), keys attached to functions but never read.

**Leave out:** weight downloads (`snapshot_download`), and provider calls that aren't billed (model
lists, billing reads, status checks). List free calls (a free moderation API) but mark them free. GPU
work with no model (a video render on a GPU host) still costs machine time: list it under machines.

## 3. Write one line per call

One line per call site and model. Split a line when one site serves several calls that are tracked
differently (two models, one covered and one not), or whose callers differ in who the call is for or
which provider they can pick; group the rest (the same agent code run with seven alike configs) into
one line with the count.

| Field | What to write |
|---|---|
| Where | `file:line` (the line that sends the call), and the function |
| Provider and model | as the code names them. A model from a variable: say where the value comes from |
| How it's called | SDK (which package), raw HTTP, through a layer (Agents SDK, LiteLLM...), or the provider is picked at runtime (from where) |
| How the result comes back | **direct**; **stream**; **queue + webhook** (name the handler); **polled** until done (Veo, a GPU host's `/status`); **written to the database** by the worker (no webhook: the finish is only seen in the task document); **inside another call** (made by one of the app's own serverless apps, or by a tool inside an agent run) |
| Who it's for | where the code knows the user and the task at that point: a user id, a task id only, or nothing (then: what called it, so it can be linked to its parent) |
| Calls per use | how many billed calls one user action can make here: retries (check retry decorators and their defaults: a retried video is a new paid video), fallbacks to another model, one call per video frame, agent turns |
| Account | the key, project, resource or region. Say whether it's a separate account or a separate key on a shared account, and which other apps or tools share it. That's how bills are told apart; say so when the bill can't split them (one key used by two code paths, a provider whose bill has no per-key split) |
| Live? | **live** (reachable from a deployed entry point); **live if config points at it** (name the config); **called from outside the code** (by a workflow built on the provider's platform, another app, a tool); **maybe dead** (say why); add **not app traffic** for scripts, tests, evals and internal tools, which are still on the same bill if they use the same key |
| Tracked | **tracked**; **partly** (the client is wrapped but this method or route isn't covered, the wrapper would record the wrong provider, as for Vertex behind an OpenAI client, or it covers only some of the providers a runtime choice can pick); **wrapper exists, not used**; **no wrapper** (name the missing piece); **no library for this language** |

**Provider by address, not by package.** An `OpenAI` client with a `base_url` pointed at Vertex,
Azure, a router or a local server is that provider. Say both.

**A provider picked at runtime:** list every provider the code can pick, and where the choice is read.
Which one runs today can't be known from code: say so.

**Is it really tracked?** Check all of these, in the repo's own code and the library version it pins:

- the wrapper covers this exact method and route (read the library's module docs or source, not only
  its README);
- the client this call uses is the wrapped one (an SDK that builds its own clients, or a subclass,
  may escape the wrapper);
- the tracker is started in that process, and its key reaches it (each function's secrets);
- the user and task are set around the call, or its parent is (for the app's own serverless apps).

## 4. List the machines the app runs

The app's own serverless and GPU apps (fal apps, Modal apps, RunPod endpoints...) cost machine time
even when nothing calls them. For each: its name, the hardware, minimum and maximum machines and how
many requests one machine takes at once (often set in CI, not the code), what calls it, the models it
runs on itself, and the calls it makes to others (listed in step 3 as "inside another call").

Also note: test and dev twins of the same app (list once, name the twins and their keys), apps still
built and deployed that nothing calls any more, and machines named in config whose code is in no repo.

## 5. Write the list

Save it to `.inference-tracker/calls.md` in the folder this was run in (create the folder; replace an
older file, keeping nothing from it). It starts with:

- the date, and each repo's branch and commit that were read (add-tracking checks them);
- **To track:** every live call that isn't fully tracked, one line each with an id (`T1`, `T2`...), its
  `file:line`, provider and model, and the library piece it needs. This is add-tracking's work list.

Then the report below. In the chat, give only the summary line and the "To track" list.

Name each app the way the company's own list of apps does (an analytics table, a config): ask the user
where that list is. A repo that isn't on it (a shared engine, a tool) is reported, but not as an app.

When a bill or a cost dashboard is at hand, check every call that looks live or untracked against it:
code that is still selectable may not have run for months. Say "dead (bill: last on <date>)" rather
than reporting it as a gap.

One report per app, in this order:

1. **Summary:** one line: live call sites, providers, how many are tracked, missing library pieces.
2. **Repos read:** each repo, the branch or tag read, commit and date. Copies and what they copy.
3. **Calls:** the table from step 3, live calls first, then "coming" (step 1), then "dead (bill: last
   on <date>)".
4. **Machines:** from step 4.
5. **Outside this folder:** backends the app calls whose code isn't here, machines with no code,
   workflows built on the provider's platform.
6. **Can't tell from code:** providers picked at runtime, models read from a database, keys that may
   also be used elsewhere, code that may or may not run.
7. **Not inference, but note it:** from step 2.
8. **Missing library pieces:** each kind of call inference-tracker can't record yet.

When run on a whole folder, start with a summary across apps: per app, the number of live call sites,
the providers, how many are tracked, and the missing library pieces, most common first. Then the
accounts shared between apps and tools, since those bills have to be split between them.

Before finishing, check the list against itself: every webhook handler has the call that sends work to
it; every serverless app has a caller (or is noted as called from outside); every key in the config and
every secret given to a function is used by a call on the list, or noted as unused; no test tool or
test copy points at production apps or keys without being noted.
