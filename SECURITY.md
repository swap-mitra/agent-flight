# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub's
[private vulnerability reporting](https://github.com/swap-mitra/ai-flight-recorder/security/advisories/new),
not in a public issue. Include what you found, how to reproduce it, and the impact you see.

You'll get an acknowledgement within a few days. Fixes are released as a new version on PyPI,
with the advisory published once users can upgrade.

## Scope

Flight Recorder writes agent runs to local files and, if you configure it, exports them to
OpenTelemetry. Issues of most interest:

- Secrets or personal data that should have been redacted reaching a cassette or a span.
- A recorded or replayed call that changes what the agent receives (interception must be invisible).
- Replay calling a write or irreversible tool live when policy forbids it.
- Any network call besides the user's own model and tool calls and their configured exporter.

## Supported versions

Only the latest release receives fixes.
