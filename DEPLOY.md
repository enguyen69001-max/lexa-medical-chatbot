# Deploy a free online demo of LEXA

This puts the **core** of LEXA online so anyone (recruiters!) can try it from a link,
**no install**: grounded medical Q&A, source citations, and the safety guards.

> The free web demo runs the offline "extractive" engine. **Voice works in the web demo**
> using the browser's built-in speech (best in **Chrome/Edge**, needs mic permission):
> say "Lexa" to wake her, and she reads answers aloud. The **conversation LLM** (local
> Ollama) is desktop-only and is simply off in the cloud — the app degrades gracefully.
>
> **Demo login:** `Freya` / `lexa-demo`

---

## Option A — Render (recommended, deploys straight from GitHub)

1. Go to **https://render.com** and **Sign up with GitHub** (free).
2. Click **New +** → **Blueprint**.
3. Select your repo **`lexa-medical-chatbot`** → **Connect**.
4. Render reads [`render.yaml`](render.yaml) automatically → click **Apply** / **Create**.
5. Wait ~2–3 min for the build. You'll get a public URL like
   `https://lexa-medical-chatbot.onrender.com`.
6. Open it, sign in (`Freya` / `lexa-demo`), and test 🎉

> Free tier note: the service "sleeps" after ~15 min of inactivity, so the **first**
> visit after a pause takes ~50 s to wake up. Subsequent loads are instant.

If the Blueprint screen is confusing, use **New + → Web Service** instead:
- Runtime: **Python**
- Build command: `pip install -r requirements.txt`
- Start command: `python medassist_app.py`
- Environment variables: `HOST=0.0.0.0`, `CHAT_LLM=off`
- Create Web Service.

---

## Option B — Hugging Face Spaces (no credit card needed)

1. Create a free account at **https://huggingface.co/join**.
2. **New → Space**. Name it `lexa`, **SDK: Docker**, visibility **Public**, **Create**.
3. In the Space's **Files** tab, **Add file → Upload files** and upload from
   `C:\PORTFOLIO AI\MEDICAL CHATBOT - LEXA\`:
   `medassist_app.py`, `requirements.txt`, `Dockerfile`,
   `logo.png`, `avatar.png`, `dash_bg.jpg`, `dash_page.jpg`.
4. Edit the Space's **README.md** (Add file → create/edit) so it starts with this header:
   ```
   ---
   title: LEXA Medical Chatbot
   emoji: 🪷
   colorFrom: green
   colorTo: yellow
   sdk: docker
   app_port: 7860
   pinned: true
   ---
   # LEXA — live demo
   Demo login: Freya / lexa-demo
   ```
5. The Space builds automatically and goes live at
   `https://huggingface.co/spaces/<your-username>/lexa`.

---

## After it's live

Send me the URL and I'll add a **🔗 Live demo** badge to the top of the main `README.md`
so recruiters see it first thing.
