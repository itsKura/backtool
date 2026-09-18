# 005 — Calling the Anthropic API over HTTP, not through the SDK

**Status:** Accepted · **Date:** 2026-09-18

## Problem

The AI layer needs the Messages API with structured outputs. The obvious choice
is the official `anthropic` Python SDK.

It cannot run on the target machine:

```
ImportError: DLL load failed while importing jiter:
An Application Control policy has blocked this file.
```

`jiter` is a Rust JSON parser that `anthropic` imports unconditionally at
package level, so `import anthropic` fails outright — not a degraded mode, not
one feature, the whole package.

## Context: this is the fourth instance

Windows Application Control on this machine has now blocked:

| Package | What broke |
| --- | --- |
| `pyarrow` | Parquet storage → [ADR 004](004-sqlite-over-parquet.md), moved to SQLite |
| Git's bundled `libcurl` | HTTPS push → switched the remote to SSH |
| Git's bundled `ssh.exe` | SSH push → pointed Git at Windows' own OpenSSH |
| `jiter` | The Anthropic SDK → this decision |

The pattern is unsigned native binaries. Anything shipping compiled extensions
is a liability in this environment, and there is no reason to expect the next
one to be different.

## Options

1. **Disable Smart App Control.** Rejected. It cannot be re-enabled without
   reinstalling Windows, and trading a machine-wide security control for a
   convenience wrapper is a bad exchange.
2. **Vendor or patch `jiter`.** Rejected: it is a transitive dependency of a
   dependency, and the fix would break on every upgrade.
3. **Call the Messages API over HTTP with `httpx`.** `httpx` is pure Python,
   already a dependency for the Binance client, and demonstrably works here.

## Decision

Option 3. `backtool.ai.client.AIClient` speaks `POST /v1/messages` directly.
`anthropic` is not a dependency of this project.

The only SDK convenience actually lost is `client.messages.parse()`, which
generates a strict JSON schema from a Pydantic model and validates the response
back into it. `strict_json_schema()` does the same job in about forty lines:

- **Inlines `$ref`/`$defs`.** Pydantic emits references for nested models;
  structured outputs cannot follow them.
- **Requires every property.** Pydantic marks only non-defaulted fields
  required.
- **Forbids extra properties** at every level, recursively.
- **Strips `default`**, which is meaningless once every field is required and
  which some validators reject alongside `required`.

Responses are validated with `Model.model_validate()` — the same Pydantic model,
the same guarantee.

## Tradeoffs

**Given up**

- Automatic retry/backoff, typed exceptions, and streaming helpers, all of which
  are reimplemented here in a smaller and less general form.
- Automatic tracking of API changes. New parameters must be added by hand.

**Gained**

- The AI layer runs at all in this environment.
- One fewer dependency, and no native code in the whole project. Every runtime
  dependency (`pandas`, `numpy`, `pydantic`, `httpx`, `python-dotenv`) is now
  either pure Python or already proven to load here.
- The request body is visible in one place, which makes the structured-output
  contract easy to inspect and test.

## Standing implication

**Prefer pure-Python dependencies for this project.** Where a package with
compiled extensions is genuinely necessary, verify it imports on the target
machine before building on it — four times is a pattern, not bad luck.

## Revisit when

The project is deployed somewhere without Application Control, *and* the SDK's
streaming or tool-runner features are actually needed. The client is small and
isolated behind `AIClient.structured()`; swapping it is a single-class change.
