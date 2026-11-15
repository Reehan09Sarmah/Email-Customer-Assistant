# Commands

Run everything from the project root.

## Setup

| Task | Command |
|---|---|
| Install dependencies | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |

## Run

| Task | Command |
|---|---|
| Start the API server (auto-reload) | `uv run fastapi dev app/main.py` |
| Check the server | open http://127.0.0.1:8000/health |
| Start the poller (auto-restart on changes) | `uv run watchfiles "uv run python -m app.poller" app .env` |
| Start the poller once, no watching | `uv run python -m app.poller` |

Run the server and the poller in separate terminals.
Use `uv run` inside the watchfiles command, otherwise it picks the wrong Python.

## Gmail

| Task | Command |
|---|---|
| Get a Gmail refresh token (one-time login) | `uv run python scripts/get_refresh_token.py` |
| Check the Gmail connection | `uv run python scripts/gmail_check.py` |

## Git

| Task | Command |
|---|---|
| See what changed | `git status` |
| Commit | `git add -A` then `git commit -m "message"` |
| Push | `git push` |
