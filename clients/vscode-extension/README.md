# HELIX VS Code extension

Inline completion (L4 FIM via `/v1/complete`) and "Ask HELIX" codebase chat (`/v1/chat`).

```bash
cd clients/vscode-extension
npm install
npm run compile
# In VS Code: Run and Debug -> "Run Extension" (F5), or package with `npm run package`
```

Settings: `helix.apiUrl` (default http://localhost:8000), `helix.devUser`, `helix.repo`, `helix.debounceMs`.

Key behaviours, matching spec 6.4:
- **Debounce** 150 ms after the last keystroke before sending.
- **Cancellation**: every new keystroke aborts the in-flight request (`AbortController`).
- **Never block typing**: any error returns no suggestion.
- **Feedback**: accepting a suggestion posts `/v1/feedback` (target `completion`) for the acceptance KPI and L4 data.
