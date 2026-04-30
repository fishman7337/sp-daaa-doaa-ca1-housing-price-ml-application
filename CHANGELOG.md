# Changelog

## 2026-04-30

- Updated project identity to EstateScope AI.
- Added MIT license, code of conduct, contribution guide, support guide, and security policy.
- Added `.env.example`, `.dockerignore`, development requirements, and CI workflow.
- Added MLOps documentation for data, model lifecycle, monitoring, and model limitations.
- Added folder-level README files for project navigation.
- Fixed Docker port default behavior for local Compose and platform deployments.
- Made TensorFlow imports lazy so Flask startup and non-ML tests are not blocked by optional NLP/CNN dependencies.
- Fixed CNN fallback training price-anchor bug.
- Pinned NumPy below 2.0 for TensorFlow compatibility.
