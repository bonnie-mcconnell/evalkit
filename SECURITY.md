# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues by emailing **security@bonnie-mcconnell.dev** with:

- A description of the vulnerability
- Steps to reproduce it
- The potential impact
- Any suggested fixes (optional)

You will receive a response within 48 hours. If the issue is confirmed, a fix
will be released as soon as possible and you will be credited in the release
notes (unless you prefer to remain anonymous).

## Scope

evalkit is a local evaluation library. The primary attack surface is:

- **Input data**: JSONL/CSV files loaded via `EvalDataset.from_jsonl()`. Malicious
  data files could trigger issues in Jinja2 template rendering. evalkit uses
  `StrictUndefined` and does not expose template rendering to untrusted input.

- **API server**: The optional FastAPI server (`evalkit[api]`) writes result files
  to `./results/`. Do not expose it to the public internet without authentication.

- **LLM provider credentials**: API keys for OpenAI/Anthropic are passed as
  environment variables and are never logged.
