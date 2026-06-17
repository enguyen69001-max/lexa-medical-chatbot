# Screenshots

Drop your screenshots here so the main `README.md` shows them. Recommended shots
(use these exact filenames and the README will pick them up automatically):

| Filename            | What to capture |
|---------------------|-----------------|
| `chat.png`          | The chat page with a **grounded answer + the source chips** (e.g. ask *"first line treatment for hypertension"*). |
| `refusal.png`       | A safety refusal (e.g. *"how much paracetamol for my 4yo"* → blocked) — great proof of the guardrails. |
| `dashboard.png`     | The dashboard page. |
| `login.png`         | The sign-in screen (optional, nice hero shot). |

## How to capture (Windows)

1. Launch the app: `dist\MedAssist.exe` (or `python medassist_app.py` → open http://127.0.0.1:8000).
2. Sign in (`Freya` / `lexa-demo`) and set up the view you want.
3. Press **Win + Shift + S**, select the window, then paste/save the image here as the
   filename above (e.g. `docs\chat.png`).
4. Re-commit: `git add docs && git commit -m "docs: add screenshots"`.

> Tip: a short screen-recording exported as a GIF (e.g. with ScreenToGif) placed at
> `docs/demo.gif` makes the repo really stand out — embed it near the top of the README.
