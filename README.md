# inferledger

Per-user inference cost for AI apps.

A small SDK records each inference call next to where it is made: who it was for, which
provider and model, the provider's request id, and the units and time it used. The provider's
bill decides how much was spent; the record decides who it was for. No prompts, images or
other user content are recorded.

Early work in progress.
