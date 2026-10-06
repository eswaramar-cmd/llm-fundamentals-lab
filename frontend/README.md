# Research Agent UI

React 19 + TypeScript + Vite front end for the LangGraph backend. Replaces the
retired static UI, which is preserved in `../frontend-legacy/`.

## Run

Two terminals, from the repository root (`llm-fundamentals-lab`).

```powershell
# terminal 1 - backend on 8000 (takes ~30-45s to warm up models)
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2 - this app on 5173
cd frontend
npm run dev
```

Open <http://localhost:5173>.

`strictPort` is on, so a busy 5173 fails loudly instead of silently moving to
5174 and leaving the backend CORS allowlist behind.

### Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Dev server on 5173 with HMR |
| `npm run build` | Typecheck, then production bundle into `dist/` |
| `npm run preview` | Serve the built bundle |
| `npm run typecheck` | `tsc` only |
| `npm run lint` | oxlint |

## Architecture

```
src/
  components/
    chat/        Feature components: panel, list, bubble, composer, timeline,
                 metrics, banner, empty state. One shared CSS module.
    layout/      App shell, header, theme toggle + connection dot.
    ui/          Dependency-free primitives: Button, Badge, Spinner.
  hooks/
    useChatStream   Owns fetch, AbortController, retry. The single source of
                    truth for stream state.
    useAutoScroll   Pins to newest output, yields when the user scrolls up.
    useTheme        Light/dark, persisted to localStorage.
    useMediaQuery   CSS media queries via useSyncExternalStore.
  lib/
    api.ts      Endpoint URLs and request-body construction.
    sse.ts      Incremental SSE frame reader.
    logger.ts   `[sse]`-prefixed console logging for every frame.
    cx.ts       Conditional class names.
    id.ts       Prefixed unique ids.
  state/
    chatReducer.ts  Pure state transitions; no I/O, so it is easy to test.
  styles/
    tokens.css   All colour, spacing, radius, and motion values.
    global.css   Reset, base element styles, reduced-motion handling.
  types/
    chat.ts    The backend wire contract. Change this with the backend.
```

Rules that keep this maintainable:

- **`useChatStream` is instantiated once, in `App`.** The header indicators and
  the chat panel read the same state. A second instance would create a second
  `AbortController` and silently desync the header from the stream.
- **Transport is split from state.** `lib/sse.ts` only turns bytes into frames;
  `chatReducer` only turns frames into state. Either can be tested alone.
- **No UI framework.** Tokens are CSS custom properties, so retheming edits one
  file and there is no runtime styling dependency.
- **`types/chat.ts` is the contract.** Field names mirror
  `backend/main.py::_sse` and `backend/agent/schemas.py::ChatRequest`.

## Backend contract

`POST /chat/stream`, request:

```json
{ "message": "...", "user_id": "default_user", "thread_id": "thread_..." }
```

`question` is still accepted as a legacy alias for `message`.

Response is `text/event-stream`. Each frame carries `type` in the data payload:

| Event | Payload | UI effect |
| --- | --- | --- |
| `status` | `message`, `node` | Status line above the composer |
| `token` | `text` | Appended to the streaming bubble |
| `tool_start` | `tool`, `input` | Adds a running row to the timeline |
| `tool_end` | `tool`, `result`, `time_ms` | Closes that row with duration |
| `approval_required` | `session_id`, `reason` | Graph paused for human approval |
| `done` | `latency_ms`, `time_to_first_token_ms`, `model`, `chunks`, `chars`, `chars_per_second`, `error`, `answer` | Freezes the bubble, fills metrics |
| `error` | `message`, `node` | Marks the bubble failed |

Throughput is reported in **characters**, not tokens. The streaming path has no
provider token usage, and counting chunks as tokens made a fast answer look like
`0.2 tokens/second` because Gemini sends a whole clause per chunk.

`done` carries `error: null` on success. When a node fails, the endpoint reads
the graph's final state so the frame carries the real failure reason and any
partial text, instead of a generic "finished without producing any text".

Frames that are not one of these are logged and skipped rather than crashing the
stream, so adding a backend event later will not break an older client.

Every frame is logged to the console with an `[sse]` prefix. Filter on that to
inspect raw traffic without touching application code.

## Configuration

Requests go to `/api` and Vite proxies them to `http://127.0.0.1:8000`
(`vite.config.ts`, override the target with the `BACKEND_ORIGIN` env var). This
keeps the browser on one origin, so development needs no CORS preflight.

To call the backend cross-origin instead, set `VITE_API_BASE_URL` in
`.env.development` (see `.env.example`). Add the origin to the backend's
`CORS_ORIGINS` if it differs from `http://localhost:5173`.

## Behaviour notes

- **Stop** aborts the request via `AbortController` and keeps the partial answer.
- **Retry** replays the last user message on the same thread.
- Token frames are coalesced and flushed once per animation frame, and message
  bubbles are memoised, so a fast provider emitting dozens of chunks per second
  does not re-render the whole transcript.
- Tool rows use native `<details>`, so they are keyboard accessible for free.
- A stream that drops mid-answer shows the banner with Retry rather than
  leaving the UI looking finished.
- Answers render as plain text with whitespace preserved. The backend emits
  Markdown (`**bold**`), so a renderer is the natural next addition.

## Performance notes

Measured on this box (CPU-only, i5-1334U, no GPU):

| Stage | Before | After |
| --- | --- | --- |
| `classify` + `retrieve_memory` | 376–985 ms | ~70 ms |
| Tokens per answer | 3 (one burst) | 18+ incremental |
| Throughput reported | `0.2 tokens/s` | `chars/s`, honest |

Application overhead is now small. Time to first token is dominated by the
provider: `gemini-3.1-flash-lite` measured 4.8–15 s TTFT with intermittent 503s
during development, while local `qwen2.5:0.5b` measured ~2.3 s. The agent node
now rotates providers on retry and falls back to local Ollama, so an outage
degrades to a slower answer instead of a failure.

`venv\Scripts\python.exe profile_stream.py "your question"` prints a per-frame
timeline against a backend on port 8010, which is the quickest way to see whether
a slow response is the graph or the provider.