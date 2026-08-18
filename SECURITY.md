# Security Policy

ISynKGR is research software. Security reports are still taken seriously, especially when they concern dependency use, unsafe file handling, command execution, data exposure, or interactions with external model endpoints.

## Supported versions

Security fixes are applied to the current `main` branch. Historical research snapshots may not receive backports.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a suspected vulnerability that could put users, systems, credentials, or data at risk.

Use GitHub's private vulnerability reporting feature for this repository when available. If private reporting is not available, contact the repository maintainers through a private channel before publishing technical details.

A useful report includes:

- affected component and revision;
- reproduction steps or proof of concept;
- expected and observed behavior;
- impact assessment;
- suggested mitigation, if known.

## Research-data guidance

Do not commit credentials, API keys, proprietary industrial data, personal data, or confidential production schemas to this repository. Example datasets and benchmark fixtures should be synthetic, public, appropriately licensed, or explicitly approved for redistribution.

## External model endpoints

LLM-backed scenarios can communicate with an Ollama endpoint configured by environment variables. Users are responsible for verifying the privacy, access-control, and network-security properties of the endpoint they select.
