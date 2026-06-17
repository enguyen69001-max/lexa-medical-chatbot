# Container image for the LEXA web demo.
# Works on any Docker host (Hugging Face Spaces, Railway, Koyeb, Fly.io, ...).
# Shows the core: grounded medical Q&A + citations + safety guards.
# (Offline voice/STT and the local conversation LLM are desktop-only.)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV HOST=0.0.0.0 \
    PORT=7860 \
    CHAT_LLM=off

EXPOSE 7860
CMD ["python", "medassist_app.py"]
