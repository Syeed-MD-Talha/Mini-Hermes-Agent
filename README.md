# Mini-Hermes Agent

A small, educational implementation of a [Hermes](https://github.com/reworkd/AgentGPT/tree/main?ref=blog.reworkd.ai)-style autonomous agent. The goal of this project is to make the core ideas behind large agent frameworks easier to understand by building a minimal but working version from scratch.

> **Status:** Work in progress. Features are being added incrementally.

## Why Mini-Hermes?

Modern agent frameworks are powerful, but their size and complexity can make them hard to learn. Mini-Hermes strips the architecture down to the essentials:

- A clean agent loop
- Persistent memory
- Tool use
- Reusable skills
- Subagent delegation

It is small enough to read in one sitting, yet complete enough to actually do useful work.

## What it can do so far

- **Conversational agent loop** powered by the Fireworks AI API through an OpenAI-compatible client.
- **Persistent memory** for durable facts about the user and environment.
- **Episodic session storage** with SQLite and FTS5 search, so the agent can recall past conversations.
- **Context compression** to keep long conversations within the model's context window.
- **Tool calling** with a growing set of built-in tools.
- **Reusable skills** that the agent can discover, load, create, and patch on the fly.
- **Subagent delegation** for isolated coding, research, review, planning, and skill-authoring tasks.
- **Workspace isolation** so generated files stay in a single `workspace/` folder.

## Architecture

```
mini_hermes/
├── agent.py              # Main agent loop and tool orchestration
├── context/              # Context compression
├── delegation/           # Subagent manager and delegate_task tool
├── learn/                # /learn mode prompt builder
├── memory/               # Curated persistent memory store
├── sessions/             # SQLite FTS5 episodic archive
├── skills/               # Skill storage, discovery, and mutation
└── tools/                # Workspace, web search, and time tools
```

## Tools

| Tool | Purpose |
|---|---|
| `memory` | Add, replace, or remove durable memory entries. |
| `session_search` | Search past conversations from the SQLite archive. |
| `skills_list` / `skill_view` | Discover and load reusable skills. |
| `skill_manage` | Create or patch skills. |
| `delegate_task` | Spawn an isolated subagent for a focused task. |
| `web_search` / `web_fetch` | Free web search via DuckDuckGo and page fetching. |
| `get_time` | Get the current time for any IANA timezone. |
| `read_file` / `write_file` / `patch` | File operations inside the workspace. |
| `search_files` | Regex search across workspace files. |
| `execute_code` | Run shell commands inside the workspace. |

## Setup

1. Clone the repository.
2. Create a `.env` file in the project root:

```bash
FIREWORKS_API_KEY=your_key_here
FIREWORKS_MODEL=accounts/fireworks/models/deepseek-v4-flash-0731
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the agent:

```bash
python run.py
```

## Usage

Type a question or command at the prompt:

```
You: What time is it in Dhaka?
You: Create a Python script that validates email addresses.
You: /learn how to deploy a FastAPI app to Railway
You: exit
```

## Roadmap

This project is intentionally incremental. Upcoming additions may include:

- Skill bundles and curated skill maintenance
- Background memory review and consolidation
- Write approval gates for destructive operations
- More robust web browsing and extraction
- Tests for the agent itself

## License

MIT
