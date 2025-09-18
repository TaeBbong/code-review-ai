# Code Review Bot (Dart + Ollama)

A lightweight CLI that reviews your staged changes using a local [Ollama](https://ollama.com) model and prints a GitHub-style code review. Includes a `pre-commit` hook installer.

## Features
- Uses your local Ollama model via HTTP
- Reviews staged git diffs (`git diff --cached`)
- GitHub-like Markdown review with per-file comments and suggestions
- Machine-readable JSON summary at the end (max severity, counts)
- Optional commit blocking on high severity
- One-command `pre-commit` hook install

## Requirements
- Dart SDK 3.3+
- Git
- Ollama running locally (default: `http://127.0.0.1:11434`)

## Install (local)
```
# From this project root
dart pub get
# Activate globally so the `code_review_bot` binary is in PATH
dart pub global activate -s path .
```
Make sure your Dart pub bin is on PATH:
- macOS/Linux: export PATH="$PATH:\$HOME/.pub-cache/bin"
- Windows (PowerShell): add `%LOCALAPPDATA%\Pub\Cache\bin` to PATH

## Usage
```
# Review staged changes (does not block)
code_review_bot review

# Use a specific model
code_review_bot review --model llama3.1:8b

# Pre-commit mode (defaults to fail on high)
code_review_bot pre-commit

# Adjust threshold
code_review_bot pre-commit --fail-on block   # only block on block
code_review_bot pre-commit --fail-on high    # default in pre-commit

# Change Ollama host/port
code_review_bot review --host 127.0.0.1 --port 11434
```

### System prompt (reviewer persona)
By default, the bot uses a system prompt to prime the model as a senior reviewer. You can override it:
```
code_review_bot review --system-prompt "You are an expert software developer who reviews many juniors' code..."
```
You may also set `OLLAMA_SYSTEM_PROMPT` in your environment.

## Install Git pre-commit hook
Inside the target Git repository:
```
code_review_bot install-hook
```
This writes `.git/hooks/pre-commit` that runs `code_review_bot pre-commit`. Make sure the `code_review_bot` executable is on PATH (via `dart pub global activate`).

## How it works
- Reads staged diff: `git diff --cached --unified=3`
- Builds a compact prompt with the diff
- Calls Ollama `/api/chat` with a system prompt (falls back to `/api/generate`)
- Prints a Markdown review and extracts a JSON summary from the model’s response
- Blocks commit if `max_severity >= --fail-on`

## Prompt contract (model output)
The bot asks the model to:
1. Print a Markdown review: high-level summary + per-file comments with "Severity: low|medium|high|block" and suggestions in fenced blocks.
2. Emit a final JSON block enclosed by ```json ... ``` with:
```
{
  "counts": {"low": n, "medium": n, "high": n, "block": n},
  "max_severity": "low|medium|high|block",
  "block": true|false
}
```
If JSON is missing, the tool treats severity as `low`.

## Tips
- For larger diffs, the tool clips to `--max-chars` (default 60000). Commit in smaller chunks for better quality.
- Try focused code models, e.g. `qwen2.5-coder:7b` or `llama3.1:8b`.
- You can export `OLLAMA_MODEL`, `OLLAMA_HOST`, `OLLAMA_PORT` to change defaults.

## Troubleshooting
- "Failed to connect to Ollama": ensure `ollama serve` is running on the configured host/port.
- No review printed: make sure you have staged changes (`git add <files>`).
- Hook does nothing: confirm `code_review_bot` is on PATH and the hook file is executable (non-Windows).

## License
MIT-like usage permitted in your environment; add your own license if publishing.
