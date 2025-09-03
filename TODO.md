# Deferred tasks (to add later)

- Code block extraction and AST validation for completions
- Safe execution harness for MBPP tests (subprocess, timeouts, import whitelist)
- Proper rubric: pass ratio over tests, detailed metrics in state
- Prompt templating (system/few-shot) and dataset-specific formatting
- Environment factories and split loaders as modules
- Result persistence to JSONL with per-test traces
- Multi-turn protocol (hint feedback, retries) with `MultiTurnEnv`
- Tool-based testing (`ToolEnv`) with `run_tests` callable
- Config flags for base URL, API key, timeouts, concurrency

Reference: Verifiers concepts and design [`https://verifiers.readthedocs.io/en/latest/overview.html`]
