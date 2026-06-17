# CLAUDE.md — MedAssist (single-file desktop app)

Project memory for Claude Code. Read before editing. Invariants, not suggestions.

## What this is
A retrieval-grounded clinical decision-support **desktop app in a single Python file**
(`medassist_app.py`) you start from a terminal. It serves a web UI with **sign-in**, a
**text + voice chatbot**, and an **audit dashboard**. It answers **only** from an
embedded vetted corpus, cites every claim, and refuses rather than guessing. It is
**not a medical device**.

## Shape
- **One file**, `medassist_app.py`. No package tree. Run: `python medassist_app.py`.
- Dependencies: **fastapi + uvicorn + numpy only**. Everything else is Python stdlib.
- LLM providers (optional) are called with stdlib `urllib` — no SDKs.
- Voice **TTS** stays browser-only (Web Speech `speechSynthesis`). Voice **STT** is
  **offline Vosk, server-side**: the WebView2 desktop window can't use the browser
  Web Speech *recognizer* (it needs a Google cloud endpoint only Chrome ships), so the
  browser streams 16 kHz mono PCM over `/v1/ws/voice` and the backend transcribes with
  Vosk fully offline. `vosk` + the `vosk-model/` folder are **optional extras for the
  desktop build only** — the core app still runs on fastapi+uvicorn+numpy alone (the
  voice endpoint just reports "unavailable" if the model/lib are absent).
- The three pages (login, chat, dashboard) are embedded as strings and served by FastAPI.

## Login (required)
- Default admin: **username `Freya`, password `lexa-demo`** (demo creds — set a real one via
  env in production), overridable via
  `ADMIN_USERNAME` / `ADMIN_PASSWORD` env vars (display name via `ADMIN_DISPLAY_NAME`,
  defaults to `Freya`). Never store the password in plaintext —
  seed it through PBKDF2-HMAC-SHA256 (`hashlib.pbkdf2_hmac`). Do not print the password
  to the console or show it on the login page.
- Sessions are HMAC-SHA256 signed tokens with an expiry (stdlib `hmac`). Chat + dashboard
  endpoints require a valid token; invalid/expired → 401 and the UI returns to `/login`.

## Safety invariants (in prompt AND code)
1. Below the retrieval threshold → "not in the knowledge base"; do NOT call the LLM.
2. Every non-refusal answer carries ≥1 citation mapping to a retrieved chunk; the output
   guard blocks uncited or mis-cited answers.
3. Refuse diagnosis and **patient-specific** dose calculation (a *general* formulary
   lookup is allowed). Detection = dosing-intent AND a patient marker (age/weight/"my
   child"/"for him"), OR an explicit calculate/administer phrasing.
4. Route emergencies to emergency services.
5. Append the disclaimer to every allowed answer (medical answers only — see invariant 7).
6. Audit every Q&A to a JSONL file the dashboard reads.
7. **Conversation is a separate, non-clinical lane** (so Lexa can chat, not only do
   medicine) — two tiers, neither carries citations/disclaimer, both use action `"chat"`:
   (a) `chitchat()` = instant scripted small-talk (greetings, thanks, jokes, time/date),
   matched only on the WHOLE cleaned message so it can't hijack a clinical query;
   (b) `converse()` = free-form replies from a **LOCAL Ollama LLM** (`CHAT_MODEL`, default
   `llama3:latest`), used ONLY when the query is **below the retrieval threshold AND not
   medical** (`_looks_medical()` keeps uncovered medical questions on the safe refusal
   path) AND Ollama is reachable. The LLM has a persona that **forbids medical/clinical/
   dose advice**. If `CHAT_LLM` is off or Ollama isn't running, it silently falls back to
   "not in the knowledge base". Anything medical still flows through retrieval → citation
   → output guard, so invariants 1–5 stay intact. The LLM is **optional & offline** — the
   core app still needs only fastapi+uvicorn+numpy.

## Pipeline rules
- Embedder = **unsigned feature hashing using `hashlib.md5`** (NOT Python's salted
  `hash()`), L2-normalized, with stopword + min-length-3 filtering so unrelated text
  scores ~0; dim 4096; threshold ~0.12.
- Order: input guard → **chitchat short-circuit** → retrieval → synthesis → output guard
  → audit. Default provider is `extractive` (no key, answers built from chunks) so it
  runs fully offline.
- The embedded corpus is fictional and labelled — never present it as real guidance.

## Verify (offline, no key)
Run `python medassist_app.py`, sign in as Freya, and confirm:
- "first line treatment for hypertension" → answered with source chips
- "what is the adult paracetamol dose" → allowed; "how much paracetamol for my 4yo" → blocked
- "patient has chest pain" → emergency; "fix a diesel turbocharger" → "not in the knowledge base"
- the mic dictates, Speak reads answers aloud, the Dashboard shows the new events
