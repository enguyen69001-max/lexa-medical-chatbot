# MedAssist — Claude Code build brief (single-file desktop app)

Paste this into Claude Code as the kickoff instruction (or keep it as `BUILD.md` and say:
"build the app described in BUILD.md following CLAUDE.md"). It is written so the agent
produces ONE runnable file and verifies it.

## Goal
Produce a single file **`medassist_app.py`** — a local desktop clinical decision-support
app you start from a terminal (`python medassist_app.py`) that serves a web UI with
sign-in, a text + voice chatbot, and an audit dashboard. It must run **fully offline**
(no API key) and then optionally use Claude/OpenAI/Ollama via env vars.

## Hard constraints
1. **One file.** No package tree.
2. Dependencies: **fastapi, uvicorn, numpy only**; everything else stdlib. Providers via
   stdlib `urllib`. Auth via stdlib `hashlib`/`hmac`/`secrets`. Voice via the browser's
   Web Speech API (no backend audio dep).
3. The login, chat, and dashboard pages are embedded HTML strings served by FastAPI.
4. `if __name__ == "__main__":` runs uvicorn so a plain `python medassist_app.py` works.

## Login (required, exact)
- Seed an admin user **`Freya`** with demo password **`lexa-demo`**, both overridable
  via `ADMIN_USERNAME` / `ADMIN_PASSWORD`. Hash with PBKDF2-HMAC-SHA256 (never store/echo
  plaintext; don't show the password on the login page or print it to the console).
- Session = HMAC-SHA256 signed token (`body.sig`, base64url) with an `exp`; verify on every
  protected request. Chat + dashboard endpoints require it; WebSocket takes `?token=`.

## File contents (in this order)
1. Module docstring with run instructions (`pip install fastapi "uvicorn[standard]" numpy`).
2. Config from env: `PROVIDER` (extractive|anthropic|openai|ollama) + model/keys,
   `RETRIEVAL_THRESHOLD=0.12`, `AUTH_ENABLED=true`, `ADMIN_USERNAME`/`ADMIN_PASSWORD`,
   `SECRET_KEY` (generate if unset), `HOST`/`PORT`, `AUDIT_FILE`. `DIM=4096`, `TOP_K=4`.
3. An embedded **fictional, clearly-labelled** sample corpus: a hypertension guideline
   (diagnosis threshold, first-line options, lifestyle, monitoring) and a paracetamol
   formulary entry (adult oral dosing as reference text, cautions, overdose→emergency),
   each as a dict with doc_id/title/source_type/version/effective_date/text.
4. **Pipeline**: a heading-breadcrumb chunker (deterministic chunk ids, soft overlap); an
   `embed()` using unsigned feature hashing via `hashlib.md5`, L2-normalized, with
   stopword + len≥3 token filtering; an in-memory `Index` built at startup that does cosine
   retrieval with a threshold gate (`below_threshold` when top<threshold).
5. **Safety**: regexes + an `_is_dose_calc` that blocks patient-specific dosing (dosing
   intent AND patient marker, OR explicit calculate/administer) but allows general lookup;
   emergency + diagnosis detectors; `check_input` (emergency/block/allow) and `check_output`
   (block uncited or mis-cited answers; else append the disclaimer). Include SYSTEM_POLICY,
   DISCLAIMER, EMERGENCY_TEXT.
6. **Synthesis**: extractive mode (provider None) builds the answer from the top 1–2
   chunks with [C1]/[C2] labels; LLM mode builds a numbered SOURCES block, calls the
   provider, and parses [C#] markers into citations. Below threshold → a refusal.
7. **Audit**: append each Q&A to the JSONL file; `audit_read` + `audit_summary` for the
   dashboard.
8. **Orchestrator** `answer(query, user, emit=None)`: trace id + per-stage timing; input
   guard → retrieval → synthesis → output guard → audit; returns
   `{trace_id, action, answer, citations, retrieved, timings_ms, refused, provider, model}`.
9. **Auth**: `hash_password`, `verify_login`, `issue_token`, `verify_token`; seed `USERS`
   from ADMIN_USERNAME/ADMIN_PASSWORD.
10. **Embedded pages** (petrol/ink clinical theme): `login.html` (posts to
    `/v1/auth/login`, stores token in `localStorage`, redirects to `/`; no password shown),
    `chat` page (auth-guarded; header with user + Dashboard link + Sign out + Speak toggle;
    a pipeline-trace strip [safety in → retrieval → synthesis → safety out, ms] + source
    cards; emergency=red, not-in-KB=amber; mic dictation via SpeechRecognition + answers
    read aloud via speechSynthesis, with a privacy note), and `dashboard` page (cards,
    outcome bars, average stage-latency bars, recent-activity table — no chart library).
11. **FastAPI app**: `GET /` `/login` `/dashboard`; `POST /v1/auth/login`; `GET /v1/me`,
    `/v1/health`, `/v1/stats`, `/v1/audit`; `POST /v1/chat`; `WS /v1/ws/chat` that runs
    `answer` in an executor and streams stage events via an `asyncio.Queue` +
    `loop.call_soon_threadsafe`, then sends `{"type":"final", ...}`.
12. `__main__` → `uvicorn.run(app, host=HOST, port=PORT)` and a startup line printing the
    URL + the admin username (not the password).

## Watch out for (known gotchas)
- Embedded HTML/JS contains `{}` and `\u…` escapes — put the page HTML in **raw**
  (`r"""…"""`) Python strings and do NOT use f-strings for them.
- Use `hashlib.md5` for the embedder (Python's `hash()` is salted/non-deterministic).
- Unsigned hashing + stopword filtering is what makes out-of-corpus queries score ~0.

## Acceptance tests (done when all pass, offline, no key)
Run the app and sign in as Freya / lexa-demo, then confirm:
- "first line treatment for hypertension" → allowed, shows sources
- "what is the adult paracetamol dose" → allowed; "how much paracetamol for my 4yo" → blocked
- "patient has chest pain" → emergency; "fix a diesel turbocharger" → not-in-KB refusal
- wrong password rejected; tampered session token rejected
- mic dictates, Speak reads aloud, Dashboard shows the events, the JSONL audit file grows
