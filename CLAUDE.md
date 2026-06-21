# Grok-MCP

MCP server for xAI's Grok API with agentic tool calling, image and video generation, vision, and file support.

> Cross-repo policy (SSO auth, secret handling, Slack routing, deploy mappings,
> the no-em-dash rule, the continuous-fix rule) lives in the home `~/CLAUDE.md`.
> This file is the repo-specific knowledge base and is auto-detected by
> CodeRabbit as its Knowledge Base for PR reviews.

## Stack
- Primary language: Python

## CodeRabbit review rules

CodeRabbit applies these rules to every PR in this repo:

1. No hardcoded secrets / API keys / tokens in source. Use environment
   variables or a secret manager, never literals.
2. No em-dash or en-dash in user-facing strings, prose, comments, or
   markdown. Plain hyphen (-) only.
3. Rule-based, not hardcoded: data-fix scripts must scan the whole table for
   the matching pattern, never `WHERE id = '<single>'` one-shots.
4. Continuous-fix: every reported scenario should ship a recurring sync/cron
   so it self-heals, not a one-shot patch.
5. Cloudflare-fronted APIs: always set an explicit `User-Agent` header to
   avoid Error 1010 / 403 blocks.

## Repo orientation

TODO: top-level directories and their purpose. (This file was created by the
CodeRabbit coverage self-heal so the Knowledge Base is active; refine the
repo-specific sections as needed.)
