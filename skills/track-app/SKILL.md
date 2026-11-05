---
name: track-app
description: Track one whole app with one command. Finds every AI call (the find-ai-calls guide), adds inference-tracker repo by repo (the add-tracking guide), opens one pull request per repo with a short description, and ends with one table of every call and where it stands. Use when asked to track an app end to end, to "do the next app", or to run find and add-tracking and open the PRs in one go.
---

# Tracking a whole app

**The goal:** one app, all its repos, from nothing to open pull requests, with one table at the end
that shows every AI call and where it stands. A person should only have to answer questions and
review the PRs.

This command adds no rules of its own about finding or tracking calls. It runs the two guides next to
it, in order, and follows them exactly:

- **find-ai-calls** (`../find-ai-calls/SKILL.md`): finds every call, saves `.inference-tracker/calls.md`.
- **add-tracking** (`../add-tracking/SKILL.md`): adds tracking to one repo, through a pull request.

Read both in full before starting (or run them as commands, if they are installed). What follows is
only the order, where to stop, and what to write at the end.

**Plain words, everywhere a person reads.** In questions, PR descriptions and the final table, name
each call by what it does ("photo edit", "avatar training", "prompt moderation"), never by its id in
`calls.md` (`T4`). Ids are fine inside `calls.md`, which is the agents' work list.

## 1. Before starting: ask once

Ask these together, in one message, then carry on:

- **Which app**, and the folder holding its repos. Its name on the company's list of apps (the
  find-ai-calls guide, step 5).
- **The app's ingest key**, or who will create it. Without it nothing is recorded, but the PRs can
  still be written: they then list the key under "Before it records anything".
- **What may be read:** the provider bills or a cost dashboard, and the production config
  (read-only). Both settle "is this live?" without asking later.
- **May it deploy to test** (the add-tracking guide, step 5)? Only with the key, and never to
  production.

## 2. Find the calls

Run the find-ai-calls guide on the app's folder, all of it, and save `.inference-tracker/calls.md`
and `calls.json`. Older files are replaced, as that guide says.

## 3. Stop: the questions only a person can answer

Ask every open question in `calls.json` (the find-ai-calls guide writes them, in one shape) **all in
one message**, each with its options and what each would mean. They cover:

- **Who a call is for**, where the code doesn't say (the add-tracking guide, step 2).
- **Code that looks dead:** does it still run? Untracked is fine for dead code; a wrong guess isn't.
- **Library changes:** each kind of call the library can't record yet. Making it means a change to
  inference-tracker (with tests, in its own change, released as a new version) before the app PRs pin
  it. Ask whether to make it, or leave the call untracked for now.
- **Which environment is test**, if the deploy files don't say.
- Calls to leave out on purpose, if any look like they should be (internal tools, free calls).

Write each answer into `calls.md` ("Left out on purpose", or the call's line) and into the
question's `answer` in `calls.json`. Then do the rest
without stopping, except for something new that only a person can decide. Never guess an answer to
keep going.

## 4. Library pieces first

For each library change the user said yes to: make it in inference-tracker as the add-tracking guide
says (section 3), release the version, then pin that version in the app PRs. Keep app and company
names out of the library.

## 5. Track, repo by repo

For each repo with calls to track, run the add-tracking guide: its own branch and worktree, its PR
into the branch the team merges into, the calls marked done in `calls.md`. One repo at a time; a repo
that is finished stays finished while the next one is done.

On the way, write down anything that wastes money, even outside the calls being tracked. These go in
the PR's "Recommendations":

- a request left running on a provider after the app stops waiting for it (a timeout, a cancelled
  task): it keeps billing;
- retries that pay again (a retried video is a new paid video), fallbacks that run two models;
- machines kept on with nothing to do (minimum machines above 0 on an app nothing calls);
- apps still deployed that nothing calls; test tools or scripts on production keys;
- one key shared by several apps or tools, so the bill can't be split.

The add-tracking guide fixes some of these itself, in their own commit (cancelling a provider's
request when the app gives up). The rest are only reported: the team decides.

## 6. Open the PR

If the key is there and test deploys are allowed, do the add-tracking guide's test run (step 5)
first. Otherwise open the PR anyway and list the test run under "Before it records anything".

Open each PR **ready to merge, not as a draft**: the goal is a PR the team can merge as it is. Don't
merge it or deploy to production yourself: the team does that. The description is short (aim for one
screen), in this shape:

```markdown
Records each AI call this repo makes with the user and task it was for, using inference-tracker
<version> (<pinned commit>). Without the key nothing is sent, and every call works exactly as before.

## Tracked
- **<call in plain words>** (<provider> <model>): for <who: the task's user, the signed-in user, no
  user (internal), or the call that made it>.

## Not tracked, and why
- **<call>**: <gone on this branch / dead (bill: last on <date>) / no library piece yet / left out on
  purpose: <reason>>.

## Before it records anything
1. Key: <who creates it>.
2. Secrets: <name>, in <each project or account the branch deploys to>, before merging.
3. Deploys: <what has to be redeployed, and from where>.

## Recommendations
- <one line each: what wastes money or could go wrong, where (`file:line`), what would fix it.
  Say if this PR already fixes it.>

## Checks
- <tests run and their result; what was checked against the real SDK with a fake network>.
```

Leave out how the library works inside, long lists of files, and anything already in the code. A
section with nothing in it is left out, except "Not tracked", which says "nothing" when nothing is
left.

## 7. The table at the end

When every repo has its PR, save one table for the whole app to `.inference-tracker/summary.md` and
print it in the chat. One row per call in `calls.md` (live, coming, dead), live ones first:

| Call | Provider and model | Where | Result comes back | Who it's for | Status | PR |
|---|---|---|---|---|---|---|
| Photo edit | fal, model from the action's config | `api/services/edit_task.py:90` | queue + webhook | the edit task's user | tracked | [api#12](link) |

- **Call:** what it does, in plain words. No ids.
- **Where:** `repo/file:line` and the function, short.
- **Result comes back:** as `calls.md` says (direct, stream, queue + webhook, polled, written to the
  database, inside another call).
- **Status:** **tracked**; **gone** (not on the branch the PR is for); **not tracked:** and why (no
  library piece yet, dead, left out on purpose, waiting on an answer).
- **PR:** the link, or "none".

Under the table, only:

- one line of totals: calls tracked, gone, not tracked;
- the PRs, one line each with its link;
- what a person has to do next, in order: keys, secrets, deploys, then the test run the add-tracking
  guide asks for (step 5) before anything ships;
- library versions made for this app, if any.
