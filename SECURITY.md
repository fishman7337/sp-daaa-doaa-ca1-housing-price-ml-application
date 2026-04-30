# Security Policy

## Supported Version

This CA repository is maintained for the AY25/26 ST1516 CA1 submission. Security fixes should target the main project copy used for assessment and demonstration.

## Reporting A Vulnerability

Do not publish secrets, exploit details, private user records, or sensitive data in public channels. Report issues to the project owner and, when required, follow Singapore Polytechnic module escalation guidance.

## Security Expectations

- Keep `.env` out of version control.
- Rotate `SECRET_KEY`, `DATABASE_URL`, and `OPENAI_API_KEY` if they are exposed.
- Use HTTPS and a managed database in production-style deployments.
- Keep user uploads constrained to image file types and size limits.
- Avoid logging raw credentials, tokens, or private prediction inputs.
- Review dependencies before upgrading model or web framework packages.

## Known Security Boundaries

- SQLite is suitable for local demonstration, not high-concurrency production.
- The chat assistant is domain-gated but should not be treated as a regulated financial adviser.
- Model predictions are estimates and should not be used as sole decision inputs.
