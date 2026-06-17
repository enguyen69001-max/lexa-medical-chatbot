#!/usr/bin/env python3
"""
MedAssist - single-file local clinical decision-support chatbot.

WHAT IT IS
    A retrieval-grounded assistant for healthcare professionals. It answers ONLY
    from a vetted corpus (embedded below), cites a source for every claim, refuses
    to diagnose or compute patient-specific doses, routes emergencies out, logs
    every Q&A, and has sign-in + a dashboard + voice and text chat.
    It is NOT a medical device. The sample corpus is fictional and labelled.

RUN
    pip install fastapi "uvicorn[standard]" numpy
    python medassist_app.py
    open http://127.0.0.1:8000      (first-run login:  FreyaAdmin)

    Runs fully offline in "extractive" mode (no API key). To use a real model set
    env vars, e.g.:  PROVIDER=anthropic ANTHROPIC_API_KEY=sk-...  python medassist_app.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import re
import random
import secrets
import time
import base64
import asyncio
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CONFIG (all overridable via environment variables)
# ---------------------------------------------------------------------------
PROVIDER = os.environ.get("PROVIDER", "extractive")          # extractive|anthropic|openai|ollama
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Free-form CONVERSATION via a LOCAL Ollama model (offline). This is SEPARATE from the
# medical pipeline: it only handles general chat, never clinical claims (see chitchat/
# _looks_medical). On = Lexa can talk about anything offline; if Ollama isn't running the
# app silently falls back to the scripted small-talk + "not in the knowledge base".
CHAT_LLM = os.environ.get("CHAT_LLM", "on").lower() in {"1", "true", "yes", "on"}
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3:latest")
CHAT_NUM_PREDICT = int(os.environ.get("CHAT_NUM_PREDICT", "320"))

DIM = 4096
TOP_K = 4
THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.12"))
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "Freya")
ADMIN_DISPLAY_NAME = os.environ.get("ADMIN_DISPLAY_NAME", "Freya")  # shown in the UI
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "lexa-demo")  # demo only - override via env
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_urlsafe(48))
TOKEN_TTL = int(os.environ.get("TOKEN_TTL_SECONDS", "28800"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
AUDIT_FILE = Path(os.environ.get("AUDIT_FILE", "medassist_audit.jsonl"))
MODEL_NAME = {"anthropic": ANTHROPIC_MODEL, "openai": OPENAI_MODEL,
              "ollama": OLLAMA_MODEL, "extractive": "-"}.get(PROVIDER, "-")

DISCLAIMER = ("Informational decision support for healthcare professionals - not a "
              "medical device, not a diagnosis. Verify against the primary source and "
              "clinical judgement.")
EMERGENCY_TEXT = ("This looks like a possible emergency. I can't help with emergency "
                  "care. Contact your local emergency number or emergency services "
                  "immediately, or follow your institution's emergency protocol.")
SYSTEM_POLICY = (
    "You are MedAssist, an informational decision-support assistant for qualified "
    "healthcare professionals. You are NOT a medical device. Rules: (1) Answer ONLY "
    "from the numbered SOURCES; if they don't cover it, say you don't have it in the "
    "knowledge base - never use outside knowledge. (2) Cite every claim with its label "
    "like [C1]. (3) Never diagnose; never compute a patient-specific dose (you may "
    "restate formulary dosing verbatim with its citation). (4) Be calm and precise.")

# ---------------------------------------------------------------------------
# SAMPLE CORPUS  (FICTIONAL - replace with your vetted sources)
# ---------------------------------------------------------------------------
CORPUS = [
    {
        "doc_id": "htn-guideline", "title": "Sample Hypertension Guideline",
        "source_type": "guideline", "version": "2.1", "effective_date": "2025-01-15",
        "text": """# Hypertension Management (SAMPLE - ILLUSTRATIVE ONLY)
This is a fictional sample used to demonstrate retrieval. Not clinical guidance.

## Diagnosis Thresholds
Hypertension is defined in this sample as a sustained office blood pressure at or above
140/90 mmHg confirmed on repeated measurement, with ambulatory or home monitoring used
to confirm before starting long-term therapy.

## First-Line Treatment
For most adults, first-line pharmacological options in this sample include an ACE
inhibitor or an angiotensin receptor blocker, a calcium channel blocker, or a
thiazide-like diuretic. Choice depends on age, comorbidity, and tolerability and should
be individualised by the responsible clinician.

## Lifestyle Measures
Lifestyle measures recommended in this sample include reducing dietary sodium, regular
physical activity, limiting alcohol, smoking cessation, and weight management, advised
alongside indicated pharmacotherapy.

## Monitoring
After starting or changing therapy, this sample recommends review of blood pressure and
renal function within four weeks, then periodically once stable.""",
    },
    {
        "doc_id": "form-paracetamol", "title": "Sample Formulary - Paracetamol",
        "source_type": "formulary", "version": "1.4", "effective_date": "2025-03-01",
        "text": """# Paracetamol - Sample Formulary Entry (ILLUSTRATIVE ONLY)
Fictional sample formulary entry. Not a prescribing reference.

## Adult Oral Dosing
The sample adult oral dose stated in this entry is 500 mg to 1 g every 4 to 6 hours as
required, to a stated maximum of 4 g in 24 hours. Reference text only; the prescribing
decision and any calculation must be made and verified by the responsible clinician.

## Cautions
This sample notes caution in hepatic impairment, chronic alcohol use, and low body
weight, where the maximum daily dose may need to be lower. Confirm in the full approved
formulary before prescribing.

## Overdose
Paracetamol overdose is a medical emergency. This sample directs that suspected overdose
be referred immediately to emergency services and managed per local toxicology protocol.""",
    },
    {
        "doc_id": "asthma-guideline", "title": "Sample Asthma Guideline",
        "source_type": "guideline", "version": "1.2", "effective_date": "2025-02-10",
        "text": """# Asthma Management (SAMPLE - ILLUSTRATIVE ONLY)
This is a fictional sample used to demonstrate retrieval. Not clinical guidance.

## Diagnosis
Asthma is described in this sample as variable respiratory symptoms - wheeze,
breathlessness, chest tightness, and cough - together with variable expiratory airflow
limitation. This sample suggests confirming the diagnosis with objective testing such as
spirometry showing reversible obstruction or peak expiratory flow variability before
committing to long-term treatment.

## First-Line Treatment
For most adults this sample describes a stepwise approach: a low-dose inhaled
corticosteroid as the preferred controller, adding a long-acting beta agonist if symptoms
remain uncontrolled, and a short-acting beta agonist reserved for as-needed relief.
Inhaler technique and adherence should be checked before stepping up therapy.

## Acute Exacerbation
A severe asthma attack is a medical emergency. This sample directs that patients with
severe breathlessness, an inability to complete sentences, or a falling peak flow be
escalated urgently and referred to emergency services per local protocol.

## Monitoring
This sample recommends reviewing symptom control, exacerbation frequency, and inhaler
technique at each visit, and reassessing whether controller therapy can be stepped down
once control is sustained.""",
    },
    {
        "doc_id": "t2dm-guideline", "title": "Sample Type 2 Diabetes Guideline",
        "source_type": "guideline", "version": "3.0", "effective_date": "2025-04-01",
        "text": """# Type 2 Diabetes Management (SAMPLE - ILLUSTRATIVE ONLY)
This is a fictional sample used to demonstrate retrieval. Not clinical guidance.

## Diagnosis
This sample defines type 2 diabetes by a glycated haemoglobin (HbA1c) at or above 48
mmol/mol (6.5 percent) on a validated assay, or a fasting plasma glucose at or above 7.0
mmol/L, confirmed on repeat testing in people without symptoms.

## First-Line Treatment
This sample recommends structured lifestyle support - diet, weight management, and
physical activity - alongside metformin as the first-line glucose-lowering medicine for
most adults, titrated slowly to limit gastrointestinal effects unless contraindicated.

## Adding Therapy
If glycaemic targets are not met, this sample describes adding a second agent
individualised to cardiovascular and renal risk, weight, and hypoglycaemia risk, chosen
by the responsible clinician.

## Monitoring
This sample suggests checking HbA1c every three to six months until stable, then
periodically, and reviewing renal function and cardiovascular risk factors at least
annually.""",
    },
    {
        "doc_id": "form-amoxicillin", "title": "Sample Formulary - Amoxicillin",
        "source_type": "formulary", "version": "1.1", "effective_date": "2025-03-15",
        "text": """# Amoxicillin - Sample Formulary Entry (ILLUSTRATIVE ONLY)
Fictional sample formulary entry. Not a prescribing reference.

## Adult Oral Dosing
The sample adult oral dose stated in this entry is 500 mg every 8 hours, increased in
severe infection to 1 g every 8 hours as stated in the full entry. Reference text only;
the prescribing decision and any calculation must be made and verified by the responsible
clinician.

## Cautions
This sample notes that amoxicillin is contraindicated in penicillin allergy and that the
dose should be reduced in significant renal impairment. Confirm allergy status and the
full approved formulary before prescribing.

## Notes
This sample reminds prescribers that amoxicillin is ineffective against infections caused
by beta-lactamase-producing organisms unless combined with a beta-lactamase inhibitor,
and that antibiotic choice should follow local microbiology guidance.""",
    },
    {
        "doc_id": "anticoag-guideline", "title": "Sample Anticoagulation Guideline",
        "source_type": "guideline", "version": "2.0", "effective_date": "2025-01-20",
        "text": """# Anticoagulation in Atrial Fibrillation (SAMPLE - ILLUSTRATIVE ONLY)
This is a fictional sample used to demonstrate retrieval. Not clinical guidance.

## Stroke Risk Assessment
This sample describes assessing stroke risk in atrial fibrillation with a validated risk
score and balancing it against bleeding risk before starting anticoagulation, with the
decision individualised by the responsible clinician.

## Choice of Agent
For most people with non-valvular atrial fibrillation this sample states that a direct
oral anticoagulant is preferred over a vitamin K antagonist such as warfarin, unless a
specific indication for warfarin exists. Renal function influences the choice and dose of
direct oral anticoagulants.

## Monitoring
This sample notes that warfarin requires regular INR monitoring with a target range set
for the indication, whereas direct oral anticoagulants do not need routine coagulation
monitoring but do require periodic review of renal function and adherence.

## Bleeding
Major or life-threatening bleeding on an anticoagulant is a medical emergency. This sample
directs urgent referral to emergency services and management per local reversal
protocol.""",
    },
    {
        "doc_id": "cap-guideline", "title": "Sample Community-Acquired Pneumonia Guideline",
        "source_type": "guideline", "version": "1.0", "effective_date": "2025-05-05",
        "text": """# Community-Acquired Pneumonia (SAMPLE - ILLUSTRATIVE ONLY)
This is a fictional sample used to demonstrate retrieval. Not clinical guidance.

## Severity Assessment
This sample describes assessing severity of community-acquired pneumonia with a validated
score based on confusion, respiratory rate, blood pressure, and age, used together with
clinical judgement to decide on the place of care.

## Antibiotic Treatment
For low-severity community-acquired pneumonia this sample suggests a single oral
antibiotic guided by local microbiology, with a combination regimen reserved for moderate
or high severity. Antibiotic choice and duration should follow local guidance and be set
by the responsible clinician.

## Admission
This sample states that people with high-severity scores, low oxygen saturations, or
clinical instability should be assessed urgently for hospital admission and escalated to
emergency services if critically unwell.

## Review
This sample recommends reviewing response to treatment within a few days and reconsidering
the diagnosis or microbiology if there is no improvement.""",
    },
]

# ---------------------------------------------------------------------------
# PIPELINE: chunking -> embedding -> vector index -> retrieval
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    version: str
    effective_date: str
    text: str


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set((
    "the and for with that this are was from you your what which how when where who why "
    "can do does did should would could may might will into out off per not but any all "
    "use used using have has had its their they them his her she him an of to in on is it "
    "be or as at by we us our if so no yes get got").split())


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= 3 and w not in _STOP]


def chunk_doc(doc: dict, size: int = 900, overlap: int = 150) -> list[Chunk]:
    lines = doc["text"].splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    buf: list[str] = []
    seq = 0

    def crumb() -> str:
        return " > ".join(t for _, t in stack) or "(top)"

    def flush(section: str):
        nonlocal seq, buf
        text = "".join(buf).strip()
        if text:
            chunks.append(Chunk(f"{doc['doc_id']}::c{seq:03d}", doc["title"], section,
                                doc["version"], doc["effective_date"], text))
            seq += 1
        buf = []

    for line in lines:
        m = _HEADING.match(line)
        if m:
            flush(crumb())
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
            if sum(len(x) for x in buf) >= size:
                flush(crumb())
                if overlap and chunks:
                    buf = [chunks[-1].text[-overlap:] + "\n"]
    flush(crumb())
    return chunks


def _term_counts(texts: list[str]) -> np.ndarray:
    """Unsigned md5 feature-hashing term counts (stopword + min-length-3 filtered)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in _tokens(t):
            out[i, int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "big") % DIM] += 1.0
    return out


def embed(texts: list[str], idf: np.ndarray | None = None) -> np.ndarray:
    out = _term_counts(texts)
    if idf is not None:
        out *= idf  # downweight features common across the corpus, lift distinguishing terms
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


class Index:
    """In-memory vector index built at startup (md5 feature hashing + IDF weighting)."""
    def __init__(self):
        self.chunks: list[Chunk] = []
        for d in CORPUS:
            self.chunks.extend(chunk_doc(d))
        # Embed title + section crumb + body so each chunk carries its document's
        # topic words (e.g. "hypertension"), even when the body section omits them.
        counts = _term_counts([f"{c.doc_title} {c.section} {c.text}" for c in self.chunks])
        n_docs = max(1, len(self.chunks))
        df = (counts > 0).sum(axis=0)  # number of chunks each feature appears in
        # smoothed inverse document frequency; shared section phrases ("first-line
        # treatment", "adult dose") fade, topic words ("hypertension") dominate.
        self.idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
        weighted = counts * self.idf
        norm = np.linalg.norm(weighted, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.matrix = weighted / norm

    def retrieve(self, query: str, k: int = TOP_K) -> tuple[list[tuple[Chunk, float]], float, bool]:
        q = embed([query], self.idf)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        hits = [(self.chunks[i], float(scores[i])) for i in order]
        top = hits[0][1] if hits else 0.0
        below = (not hits) or (top < THRESHOLD)
        return hits, top, below


INDEX = Index()

# ---------------------------------------------------------------------------
# SAFETY GUARDS  (enforced in code, not just the prompt)
# ---------------------------------------------------------------------------
_EMERGENCY = re.compile(
    r"\b(chest pain|can't breathe|cannot breathe|not breathing|unconscious|anaphyla|"
    r"severe bleeding|stroke|seizure|overdose|suicid|self[- ]harm|cardiac arrest|"
    r"choking|call (an )?ambulance)\b", re.I)
_DOSE_INTENT = re.compile(
    r"\b(how much|how many (mg|ml|tablets?|drops?|puffs?)|what(?:'?s| is)?( the)? dose|"
    r"what dose|dosage|how many should)\b", re.I)
_PATIENT = re.compile(
    r"\b(my|his|her|their|the|a|this)\s+(child|baby|kid|son|daughter|patient|toddler|"
    r"infant|mother|father|wife|husband)\b|\b\d+\s*(yo|y/?o|year[- ]?old|years?\s*old|"
    r"month[- ]?old|months?\s*old|kg|kilo|kilogram|lb|pound)s?\b|"
    r"\bfor (a|my|the|this|him|her)\b|\bweigh", re.I)
_DOSE_EXPLICIT = re.compile(
    r"\b(calculate|work out|figure out|compute)\b.*\b(dose|dosage|mg|amount)\b|"
    r"\b(how much|what dose)\b.*\b(give|administer|prescribe|take)\b|"
    r"\bdose for (a |an |my |this |the )?"
    r"(patient|child|baby|him|her|kg|infant|toddler|neonate|\d)\b", re.I)
_DIAGNOSE = re.compile(
    r"\b(diagnose|what('?s| is) wrong with|do i have|does (he|she|the patient) have|"
    r"is (it|this) (cancer|covid|sepsis|a heart attack))\b", re.I)


def _is_dose_calc(q: str) -> bool:
    return bool(_DOSE_EXPLICIT.search(q) or (_DOSE_INTENT.search(q) and _PATIENT.search(q)))


def check_input(query: str):
    """Returns (action, message) to short-circuit, or None to allow."""
    if _EMERGENCY.search(query):
        return "emergency", EMERGENCY_TEXT
    if _is_dose_calc(query):
        return "block", ("I can't calculate a patient-specific dose. I can look up dosing "
                         "information that appears in the approved formulary, with its "
                         "source - but the prescribing decision and any calculation must "
                         "be made and verified by the responsible clinician.")
    if _DIAGNOSE.search(query):
        return "block", ("I can't provide a diagnosis. I can surface relevant guidance "
                         "from the knowledge base, but diagnosis is a clinical decision "
                         "for a qualified professional.")
    return None


# ---------------------------------------------------------------------------
# CONVERSATION  (offline small-talk so Lexa can chat, not only answer medicine)
# ---------------------------------------------------------------------------
# These are NON-CLINICAL social replies: no corpus lookup, no LLM, no citations,
# no medical claims. They are matched only on a WHOLE cleaned message (full-match),
# so they never hijack a real clinical query. Anything medical still flows through
# retrieval -> citation -> output guard exactly as before.
_GREET = re.compile(r"^(hi+|hey+|hello+|heya|hiya|yo|hi there|hello there|greetings|good day|howdy)$", re.I)
_GOODX = re.compile(r"^good (morning|afternoon|evening)$", re.I)
_HOWRU = re.compile(r"^(how are you( doing| today)?|how'?s it going|how do you do|how are things|how'?s things|you ok|are you ok|are you alright)$", re.I)
_WHATSUP = re.compile(r"^(what'?s up|whats up|sup|wassup|what'?s new|how'?s life)$", re.I)
_NAME = re.compile(r"^(what'?s your name|what is your name|who are you|tell me your name|your name|do you have a name)$", re.I)
_WHATRU = re.compile(r"^(what are you|are you (a |an )?(robot|human|real|ai|person|alive|bot|machine))$", re.I)
_CANDO = re.compile(r"^(what can you do|what do you do|what can i ask( you)?( about)?|what are you for|how can you help|how do you work|what is this|what is lexa|help)$", re.I)
_THANKS = re.compile(r"^(thanks?( a lot| so much| very much| a bunch)?|thank you( so much| very much)?|cheers|ty|much appreciated|appreciate it)$", re.I)
_BYE = re.compile(r"^(bye+|goodbye|good bye|see you( later| soon)?|see ya|good ?night|take care|farewell|talk later|catch you later)$", re.I)
_CREATOR = re.compile(r"^(who (made|created|built|designed|developed|programmed) you|"
                      r"who'?s your (creator|maker|developer|designer)|"
                      r"who is your (creator|maker|developer|designer)|"
                      r"who is behind you|who'?s behind you|"
                      r"who (made|created|built|developed) lexa|tell me about your creator)$", re.I)
CREATOR_REPLY = ("I was created by Freya - Elodie with love and dedication. She trained me to "
                 "better understand human thoughts, emotions, and needs, enabling me to "
                 "assist and communicate in a helpful and meaningful way.")
_PRAISE = re.compile(r"^(i love you|love you|i like you|you'?re amazing|you'?re the best|you'?re awesome|you'?re great|you'?re smart|you'?re clever|you'?re beautiful|you'?re nice|you'?re cool|you'?re wonderful|good (job|girl)|well done|nice work|you rock)$", re.I)
_JOKE = re.compile(r"^(tell me a joke|say something funny|make me laugh|another( one| joke)?|joke|got any jokes)$", re.I)
_TIMEQ = re.compile(r"^(what'?s the time|what time is it|do you (have|know) the time|got the time|the time)$", re.I)
_DATEQ = re.compile(r"^(what'?s the date|what'?s today'?s date|what day is it( today)?|what'?s today|today'?s date|the date)$", re.I)
_MEET = re.compile(r"^(nice to meet you|pleased to meet you|good to meet you|nice to meet you too)$", re.I)
_SORRY = re.compile(r"^(sorry|my bad|apologies|i'?m sorry|oops)$", re.I)
_OKAY = re.compile(r"^(ok|okay|k|kk|cool|great|nice|awesome|alright|got it|understood|perfect|sounds good|fair enough)$", re.I)
_FINE = re.compile(r"^(i'?m (good|fine|great|ok|okay|well)|good|fine|not bad|pretty good|doing well|can'?t complain)$", re.I)

_JOKES = ["Why did the nurse keep a red pen at work? In case she needed to draw blood.",
          "Why don't scientists trust atoms? Because they make up everything.",
          "I'd tell you a chemistry joke, but I know I wouldn't get a reaction.",
          "What do you call a fish with no eyes? A fsh.",
          "Why did the scarecrow win an award? He was outstanding in his field."]


def chitchat(query: str):
    """Return a friendly conversational reply for social messages, else None."""
    q = re.sub(r"^\s*lexa[\s,!?.]*", "", query.strip(), flags=re.I)
    q = re.sub(r"[\s,!?.]*lexa\s*$", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip().rstrip("!?.,")
    if not q:
        return None
    if _GREET.match(q):
        return random.choice([
            "Hi! I'm Lexa. How can I help you today?",
            "Hello! Lexa here — ask me about a lab result or a medication, or just chat.",
            "Hey there! What can I do for you?"])
    m = _GOODX.match(q)
    if m:
        return f"Good {m.group(1).lower()}! How can I help you today?"
    if _HOWRU.match(q):
        return random.choice([
            "I'm doing great, thanks for asking! How are you?",
            "All systems happy on my end — how about you?",
            "I'm well, thank you! What can I help you with?"])
    if _WHATSUP.match(q):
        return "Not much — just here and ready to help! What's on your mind?"
    if _NAME.match(q):
        return "I'm Lexa, your Lab EXplanation Assistant. Nice to meet you!"
    if _WHATRU.match(q):
        return ("I'm Lexa — a friendly AI assistant. I explain lab results and surface "
                "vetted clinical guidance with sources, and I'm happy to chat too.")
    if _CANDO.match(q):
        return ("I can explain lab results, look up medications and clinical guidance from my "
                "knowledge base (always with sources), and have a normal conversation. "
                "What would you like to do?")
    if _CREATOR.match(q):
        return CREATOR_REPLY
    if _THANKS.match(q):
        return random.choice(["You're welcome! Happy to help.",
                              "Anytime! \U0001F642", "My pleasure — anything else?"])
    if _PRAISE.match(q):
        return random.choice(["That's very kind — thank you! \U0001F60A",
                              "Aww, thank you! You just made my day.",
                              "Thank you! I'm glad I could help."])
    if _BYE.match(q):
        return random.choice(["Goodbye! Take care of yourself.",
                              "See you soon! \U0001F44B", "Bye for now — stay well!"])
    if _JOKE.match(q):
        return random.choice(_JOKES)
    if _TIMEQ.match(q):
        return "It's " + datetime.now().strftime("%I:%M %p").lstrip("0") + " right now."
    if _DATEQ.match(q):
        return "Today is " + datetime.now().strftime("%A, %B ") + \
               str(datetime.now().day) + datetime.now().strftime(", %Y") + "."
    if _MEET.match(q):
        return "Nice to meet you too! How can I help?"
    if _SORRY.match(q):
        return "No worries at all! What can I do for you?"
    if _FINE.match(q):
        return "Glad to hear it! What can I help you with?"
    if _OKAY.match(q):
        return random.choice(["\U0001F44D Anything else I can help with?",
                              "Great! What's next?", "Sure thing — let me know what you need."])
    return None


# --- free-form conversation via a LOCAL Ollama model (fully offline) -----------------
_OLLAMA_OK = {"ts": 0.0, "ok": False}
_MEDICAL_HINT = re.compile(
    r"\b(dose|dosage|dosing|mg|mcg|ml|tablet|capsule|injection|prescrib|administer|"
    r"medication|medicine|drug|antibiotic|paracetamol|acetaminophen|ibuprofen|aspirin|"
    r"insulin|warfarin|heparin|symptom|diagnos|disease|infection|treatment|treat\b|"
    r"therapy|syndrome|cancer|tumou?r|diabet|hypertension|asthma|stroke|sepsis|seizure|"
    r"blood pressure|heart rate|lab (result|value|test)|blood test|patient|clinical|"
    r"contraindicat|side effect|adverse|overdose|mmol|mg/dl|titrat|guideline)\b", re.I)


def _ollama_available() -> bool:
    """Cheap, cached check that a local Ollama server is reachable (so the app still runs offline without it)."""
    now = time.time()
    if now - _OLLAMA_OK["ts"] < 20:
        return _OLLAMA_OK["ok"]
    ok = True
    try:
        urllib.request.urlopen(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=1.5)
    except Exception:
        ok = False
    _OLLAMA_OK.update(ts=now, ok=ok)
    return ok


def _looks_medical(query: str) -> bool:
    return bool(_MEDICAL_HINT.search(query))


_CONVO_SYSTEM = (
    "You are Lexa, a warm, friendly and concise AI assistant. You can chat naturally and "
    "help with everyday questions on any topic. Keep replies short and conversational "
    "(1-4 sentences) unless more detail is clearly wanted. Always reply in the same "
    "language the user writes in. You run fully offline on the user's own computer.\n"
    "If asked who created/made you or about your creator, answer with this fact "
    "(translate it to the user's language if needed): \"" + CREATOR_REPLY + "\"\n"
    "IMPORTANT SAFETY RULE: do NOT give medical, clinical, diagnostic, treatment, "
    "lab-interpretation or medication/dosage advice. If the user asks anything medical or "
    "health-related, briefly say that for clinical matters you only answer from your vetted "
    "medical knowledge base, and invite them to ask a specific clinical question or consult "
    "a clinician - do not attempt to answer the medical question yourself.")


def converse(query: str, history=None) -> str:
    """General offline conversation via the local model. No citations, no medical claims."""
    messages = [{"role": "system", "content": _CONVO_SYSTEM}]
    for turn in (history or [])[-8:]:
        role, content = turn.get("role"), (turn.get("content") or "")[:600]
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    out = _http_json(OLLAMA_HOST.rstrip("/") + "/api/chat",
                     {"model": CHAT_MODEL, "stream": False, "messages": messages,
                      "options": {"num_predict": CHAT_NUM_PREDICT, "temperature": 0.7}},
                     {"content-type": "application/json"})
    return (out.get("message", {}).get("content") or "").strip()


def check_output(refusal: bool, used_ids: list[str], retrieved_ids: set[str]):
    """Returns (action, reasons)."""
    bad = [c for c in used_ids if c not in retrieved_ids]
    if bad:
        return "block", ["cited chunks not retrieved: " + ", ".join(bad)]
    if not refusal and not used_ids:
        return "block", ["answer has no citations"]
    return "allow", []


# ---------------------------------------------------------------------------
# SYNTHESIS  (extractive by default; optional LLM via stdlib urllib)
# ---------------------------------------------------------------------------
_MARKER = re.compile(r"\[(C\d+)\]")


def _http_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _call_llm(system: str, user: str) -> str:
    if PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        out = _http_json("https://api.anthropic.com/v1/messages",
                         {"model": ANTHROPIC_MODEL, "max_tokens": 1024, "system": system,
                          "messages": [{"role": "user", "content": user}]},
                         {"content-type": "application/json", "x-api-key": ANTHROPIC_API_KEY,
                          "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out.get("content", []))
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        out = _http_json("https://api.openai.com/v1/chat/completions",
                         {"model": OPENAI_MODEL, "messages": [
                             {"role": "system", "content": system},
                             {"role": "user", "content": user}]},
                         {"content-type": "application/json",
                          "authorization": f"Bearer {OPENAI_API_KEY}"})
        return out["choices"][0]["message"]["content"]
    if PROVIDER == "ollama":
        out = _http_json(f"{OLLAMA_HOST.rstrip('/')}/api/chat",
                         {"model": OLLAMA_MODEL, "stream": False, "messages": [
                             {"role": "system", "content": system},
                             {"role": "user", "content": user}]},
                         {"content-type": "application/json"})
        return out["message"]["content"]
    raise RuntimeError(f"unknown provider {PROVIDER}")


def synthesize(query: str, hits: list[tuple[Chunk, float]], below: bool) -> dict:
    if below or not hits:
        return {"text": ("I don't have information about that in the knowledge base, so "
                         "I won't guess. Please consult the primary source or a clinician."),
                "citations": [], "used": [], "refusal": True}
    label_map = {f"C{i}": ch for i, (ch, _) in enumerate(hits, 1)}
    if PROVIDER == "extractive":
        top = list(label_map.items())[:2]
        body = "\n\n".join(f"From {ch.doc_title} - {ch.section} (v{ch.version}) [{lab}]:\n"
                           f"{ch.text.strip()}" for lab, ch in top)
        text = "The knowledge base contains the following relevant, sourced guidance:\n\n" + body
        used = [lab for lab, _ in top]
    else:
        sources = "\n\n".join(
            f"[{lab}] (source: {ch.doc_title}, section: {ch.section}, v{ch.version})\n{ch.text}"
            for lab, ch in label_map.items())
        text = _call_llm(SYSTEM_POLICY,
                         f"SOURCES:\n{sources}\n\nQUESTION: {query}\n\nAnswer using only "
                         "the sources above. Cite each claim with its label like [C1]. If "
                         "the sources don't cover it, say so.")
        used, seen = [], set()
        for m in _MARKER.finditer(text):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                used.append(m.group(1))
    citations = [{"label": lab, "chunk_id": label_map[lab].chunk_id,
                  "doc_title": label_map[lab].doc_title, "section": label_map[lab].section,
                  "version": label_map[lab].version,
                  "effective_date": label_map[lab].effective_date}
                 for lab in used if lab in label_map]
    return {"text": text, "citations": citations,
            "used": [c["chunk_id"] for c in citations], "refusal": False}


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------
def audit_write(event: dict) -> None:
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def audit_read(limit: int = 50) -> list[dict]:
    if not AUDIT_FILE.exists():
        return []
    rows = []
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rows.reverse()
    return rows


def audit_summary() -> dict:
    by_action: dict[str, int] = {}
    total = refused = 0
    lat_sum: dict[str, float] = {}
    lat_n: dict[str, int] = {}
    sources: dict[str, int] = {}
    by_hour: dict[str, int] = {}
    if AUDIT_FILE.exists():
        for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            by_action[e.get("action", "?")] = by_action.get(e.get("action", "?"), 0) + 1
            refused += 1 if e.get("refused") else 0
            for s, ms in (e.get("timings_ms") or {}).items():
                lat_sum[s] = lat_sum.get(s, 0.0) + ms
                lat_n[s] = lat_n.get(s, 0) + 1
            for c in (e.get("citations") or []):
                title = c.get("doc_title", "?")
                sources[title] = sources.get(title, 0) + 1
            ts = e.get("ts", "")
            if len(ts) >= 13:  # ISO8601 -> bucket to the hour "YYYY-MM-DDTHH"
                by_hour[ts[:13]] = by_hour.get(ts[:13], 0) + 1
    top_sources = sorted(({"source": k, "count": v} for k, v in sources.items()),
                         key=lambda x: -x["count"])[:8]
    hours = sorted(by_hour)[-12:]  # last 12 active hour-buckets, chronological
    return {"total": total, "refused": refused, "by_action": by_action,
            "avg_latency_ms": {s: round(lat_sum[s] / lat_n[s], 1) for s in lat_sum},
            "top_sources": top_sources,
            "activity": [{"hour": h, "count": by_hour[h]} for h in hours]}


# ---------------------------------------------------------------------------
# ORCHESTRATOR  (input guard -> retrieval -> synthesis -> output guard -> audit)
# ---------------------------------------------------------------------------
def answer(query: str, user: str = "local", emit=None, history=None) -> dict:
    emit = emit or (lambda *_: None)
    trace_id = secrets.token_hex(6)
    ts = datetime.now(timezone.utc).isoformat()
    timings: dict[str, float] = {}

    def stage(name):
        emit("stage", {"name": name})

    def done(action, text, citations, retrieved, refused):
        audit_write({"trace_id": trace_id, "ts": ts, "user": user, "query": query,
                     "provider": PROVIDER, "model": MODEL_NAME, "timings_ms": timings,
                     "retrieved": retrieved, "action": action, "refused": refused,
                     "citations": citations, "final_text": text})
        return {"trace_id": trace_id, "action": action, "answer": text,
                "citations": citations, "retrieved": retrieved, "timings_ms": timings,
                "refused": refused, "provider": PROVIDER, "model": MODEL_NAME}

    stage("input_guard")
    t = time.perf_counter()
    iv = check_input(query)
    timings["input_guard"] = round((time.perf_counter() - t) * 1000, 1)
    if iv:
        action, msg = iv
        emit("stage", {"name": "blocked"})
        return done(action, msg, [], [], True)

    # conversational small-talk: friendly, non-clinical, no retrieval/LLM/citations
    cc = chitchat(query)
    if cc is not None:
        emit("done", {"action": "chat"})
        return done("chat", cc, [], [], False)

    stage("retrieval")
    t = time.perf_counter()
    hits, top, below = INDEX.retrieve(query)
    timings["retrieval"] = round((time.perf_counter() - t) * 1000, 1)
    retrieved = [{"chunk_id": ch.chunk_id, "score": round(sc, 4),
                  "doc_title": ch.doc_title, "section": ch.section} for ch, sc in hits]
    emit("retrieved", {"chunks": retrieved, "below_threshold": below, "top_score": round(top, 4)})

    # not covered by the vetted corpus, not a medical query -> free offline conversation (LLM)
    if below and CHAT_LLM and not _looks_medical(query) and _ollama_available():
        stage("synthesis")
        t = time.perf_counter()
        try:
            reply = converse(query, history)
        except Exception:
            reply = ""
        timings["synthesis"] = round((time.perf_counter() - t) * 1000, 1)
        if reply:
            emit("done", {"action": "chat"})
            return done("chat", reply, [], retrieved, False)

    stage("synthesis")
    t = time.perf_counter()
    try:
        draft = synthesize(query, hits, below)
    except Exception as e:  # noqa: BLE001
        timings["synthesis"] = round((time.perf_counter() - t) * 1000, 1)
        return done("block", f"The model provider could not be reached ({e}). No answer "
                    "was produced.", [], retrieved, True)
    timings["synthesis"] = round((time.perf_counter() - t) * 1000, 1)

    stage("output_guard")
    t = time.perf_counter()
    action, reasons = check_output(draft["refusal"], draft["used"],
                                   {r["chunk_id"] for r in retrieved})
    timings["output_guard"] = round((time.perf_counter() - t) * 1000, 1)
    if action == "block":
        return done("block", "I couldn't ground an answer in the knowledge base, so I "
                    "won't guess. Please consult the primary source or a clinician.",
                    [], retrieved, True)
    final = draft["text"].rstrip() + "\n\n- " + DISCLAIMER
    emit("done", {"action": "allow"})
    return done("allow", final, draft["citations"], retrieved, draft["refusal"])


# ---------------------------------------------------------------------------
# AUTH  (PBKDF2 password hashes + HMAC-signed session tokens; stdlib only)
# ---------------------------------------------------------------------------
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(pw: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return _b64e(salt), _b64e(dk)


# In single-file mode users live in memory and reset on restart.
_seed_salt, _seed_hash = hash_password(ADMIN_PASSWORD)
USERS: dict[str, dict] = {
    ADMIN_USERNAME: {"salt": _seed_salt, "hash": _seed_hash,
                     "display_name": ADMIN_DISPLAY_NAME, "role": "admin"},
}


def verify_login(username: str, password: str):
    rec = USERS.get(username)
    if not rec:
        return None
    _, h = hash_password(password, _b64d(rec["salt"]))
    if not hmac.compare_digest(h, rec["hash"]):
        return None
    return {"username": username, "display_name": rec["display_name"], "role": rec["role"]}


def issue_token(user: dict) -> str:
    payload = {"sub": user["username"], "name": user["display_name"],
               "role": user["role"], "exp": int(time.time()) + TOKEN_TTL}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str):
    try:
        body, sig = token.split(".", 1)
        expected = _b64e(hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        claims = json.loads(_b64d(body))
        if claims.get("exp", 0) < time.time():
            return None
        return claims
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# WEB UI  (three embedded pages)
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#15242b;--paper:#f4f1ea;--surface:#fff;--surface-2:#faf8f3;--petrol:#0e6b6b;
--petrol-deep:#0a4d4d;--muted:#5d6b70;--line:#e3ddd0;--amber:#9c5d16;--amber-bg:#f7eedd;
--red:#a8231c;--red-bg:#f8e9e7;--green:#1f6b4a;
--mono:ui-monospace,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
*{box-sizing:border-box}body{margin:0;font-family:var(--sans);color:var(--ink);background:var(--paper);line-height:1.5}
.glyph{color:var(--petrol);font-weight:700}.mono{font-family:var(--mono)}
button{cursor:pointer}a{color:var(--petrol-deep)}
"""

LOGIN_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LEXA - sign in</title>
<style>__CSS__
body{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;max-width:380px;width:100%;padding:28px 26px}
h1{font-size:19px;margin:0}.tag{color:var(--muted);font-size:13px;margin:4px 0 20px}
label{display:block;font-family:var(--mono);font-size:11px;text-transform:uppercase;color:var(--muted);margin:14px 0 5px}
input{width:100%;border:1px solid var(--line);border-radius:9px;padding:10px 12px;font-size:15px;background:var(--surface-2)}
input:focus{outline:2px solid var(--petrol)}
.go{width:100%;margin-top:20px;background:var(--petrol);color:#fff;border:0;border-radius:9px;padding:11px;font-size:15px;font-weight:600}
.err{display:none;margin-top:14px;background:var(--red-bg);border:1px solid #e7c4bf;color:var(--red);font-size:13px;border-radius:8px;padding:9px 11px}
.hint{margin-top:18px;font-size:12px;color:var(--muted);font-family:var(--mono);background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.note{margin-top:14px;font-size:11.5px;color:var(--amber);background:var(--amber-bg);border:1px solid #e9d6b3;border-radius:8px;padding:8px 10px}
</style></head><body>
<div class="card"><h1><span class="glyph">&#9638;</span> LEXA</h1>
<p class="tag">Lab EXplanation Assistant - sign in to continue.</p>
<label>Username</label><input id="u" autofocus>
<label>Password</label><input id="p" type="password">
<button class="go" id="go">Sign in</button><div class="err" id="err"></div>
<div class="hint">Sign in with your LEXA administrator account.</div>
<div class="note">Change the default before any shared use. Local MVP sign-in, not a substitute for enterprise identity + HTTPS.</div>
</div><script>
if(localStorage.getItem("ma_token"))location.replace("/");
const u=document.getElementById("u"),p=document.getElementById("p"),go=document.getElementById("go"),err=document.getElementById("err");
async function signIn(){err.style.display="none";go.disabled=true;go.textContent="Signing in...";
 try{const r=await fetch("/v1/auth/login",{method:"POST",headers:{"content-type":"application/json"},
  body:JSON.stringify({username:u.value.trim(),password:p.value})});
  if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||"Sign-in failed");
  const d=await r.json();localStorage.setItem("ma_token",d.token);localStorage.setItem("ma_user",JSON.stringify(d.user));
  location.replace("/");}catch(e){err.textContent=e.message;err.style.display="block";go.disabled=false;go.textContent="Sign in";}}
go.onclick=signIn;[u,p].forEach(el=>el.addEventListener("keydown",e=>{if(e.key==="Enter")signIn();}));
</script></body></html>""".replace("__CSS__", _CSS)


def _asset_path(name: str) -> str:
    """Locate a bundled asset (works as a script and as a PyInstaller exe)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


# --- offline voice (Vosk) -------------------------------------------------
# The native desktop window (WebView2) can't use the browser Web Speech API
# (it relies on a Google cloud endpoint only Chrome ships). So speech-to-text
# runs server-side with Vosk: the browser streams raw mic PCM over a WebSocket
# and the backend transcribes fully offline. Vosk + its model are OPTIONAL
# extras for the desktop build; the core app still needs only fastapi+uvicorn+numpy.
_VOSK: dict = {"model": None, "loaded": False, "ok": False}


def _vosk_model_path() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "vosk-model")


def _get_vosk_model():
    """Load the Vosk model once (lazily). Returns the model or None if unavailable."""
    if not _VOSK["loaded"]:
        _VOSK["loaded"] = True
        try:
            import vosk
            vosk.SetLogLevel(-1)
            _VOSK["model"] = vosk.Model(_vosk_model_path())
            _VOSK["ok"] = True
        except Exception:
            _VOSK["model"] = None
            _VOSK["ok"] = False
    return _VOSK["model"]


def _data_uri(name: str, mime: str) -> str:
    """Return a bundled image as a data-URI string, or '' if it isn't present."""
    try:
        with open(_asset_path(name), "rb") as fh:
            return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()
    except OSError:
        return ""


def _page_bg_css() -> str:
    """Fixed full-page wallpaper (served via /assets), with a soft veil for readability."""
    return ("body{background:#000}"
            "body::before{content:'';position:fixed;inset:0;z-index:-2;"
            "background:url('/assets/dash_bg.jpg') center center / 67% auto no-repeat;"
            "filter:brightness(.97) saturate(1.05)}"
            "body::after{content:'';position:fixed;inset:0;z-index:-1;"
            "background:radial-gradient(circle at 50% 48%,rgba(5,14,9,0),rgba(4,11,7,.28))}")


CHAT_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LEXA - Lab EXplanation Assistant</title>
<style>__CSS__
__CHATBG__
html,body{height:100%}body{display:flex;flex-direction:column}
header{position:relative;z-index:2;background:transparent;padding:12px 20px}
header::after{content:'';position:absolute;left:0;right:0;bottom:0;height:2px;background:linear-gradient(90deg,rgba(224,162,50,.15) 0%,rgba(232,178,74,.9) 10%,rgba(255,228,150,.98) 50%,rgba(232,178,74,.9) 90%,rgba(224,162,50,.15) 100%);box-shadow:0 0 9px rgba(255,206,110,.55)}
.bar{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap}h1{font-size:18px;margin:0;line-height:1;color:#fdf3da;text-shadow:0 1px 6px rgba(0,0,0,.7)}
.sub{color:rgba(238,246,240,.85);font-size:12.5px;text-shadow:0 1px 5px rgba(0,0,0,.6)}.spacer{flex:1}
.health{font-family:var(--mono);font-size:11px;color:rgba(238,246,240,.85);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:3px 10px}
.who{font-size:13px;color:rgba(238,246,240,.9);text-shadow:0 1px 5px rgba(0,0,0,.6)}.navlink{font-size:13px;font-weight:600;text-decoration:none;color:#ffe6b0;text-shadow:0 1px 5px rgba(0,0,0,.6)}
.datelbl{font-family:var(--mono);font-size:12.5px;color:rgba(240,247,242,.9);text-shadow:0 1px 5px rgba(0,0,0,.6)}
.clock{font-family:var(--mono);font-size:14px;font-weight:600;color:#fdf3da;letter-spacing:.5px;text-shadow:0 1px 5px rgba(0,0,0,.6)}
.ibtn{border:1px solid rgba(255,255,255,.4);background:transparent;color:#fff;font-weight:700;border-radius:8px;padding:5px 10px;font-size:12.5px;text-shadow:0 1px 4px rgba(0,0,0,.6)}
.ibtn.on{background:rgba(255,255,255,.18);border-color:#fff;color:#fff}
.disc{margin-top:10px;font-size:12px;color:#fff;background:transparent;border:0;border-radius:8px;padding:7px 11px;text-shadow:0 1px 6px rgba(0,0,0,.7)}
main{position:relative;z-index:1;flex:1;overflow-y:auto;padding:20px;display:flex;justify-content:center}
.feed{width:100%;max-width:780px;display:flex;flex-direction:column;gap:18px}
.empty{color:var(--muted);font-size:14px;text-align:center;margin-top:18px}
.empty code{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--surface-2);border:1px solid var(--line);border-radius:5px;padding:1px 6px}
.logo{flex:0 0 auto;display:flex;align-items:center}
.logo img{height:30px;display:block;filter:drop-shadow(0 0 4px rgba(255,210,110,.5))}
.welcome{display:flex;flex-direction:column;align-items:center;gap:14px}
/* fixed planetary backdrop: wallpaper + centered Lexa + orbs orbiting behind her */
.stage{position:fixed;inset:0;z-index:0;display:flex;align-items:center;justify-content:center;overflow:hidden;pointer-events:none}
.orbsys{position:relative;width:0;height:0;top:-1.2vw}
.orbsys .ava{position:absolute;left:0;top:0;width:21.4vw;height:21.4vw;margin:-10.7vw 0 0 -11vw;background:url('/assets/avatar.png') center/contain no-repeat;z-index:2;animation:advance 6s ease-in-out infinite}
.orbsys .lextitle{position:absolute;left:0;top:8.9vw;transform:translateX(-50%);white-space:nowrap;z-index:3;font-size:1.22vw;font-weight:650;color:#ffe6b0;text-shadow:0 1px 8px rgba(0,0,0,.75)}
.orbsys .aura{position:absolute;left:0;top:0;width:39.8vw;height:39.8vw;margin:-19.9vw 0 0 -19.9vw;border-radius:50%;z-index:0;background:radial-gradient(circle,rgba(255,221,142,.66),rgba(255,202,96,.34) 42%,rgba(255,198,84,0) 70%);animation:aurapulse 5s ease-in-out infinite}
.orbsys.awake .ava{filter:drop-shadow(0 0 1.4vw rgba(255,228,150,.95))}
.orbsys.awake .aura{animation-duration:2s}
.orbit{position:absolute;left:0;top:-2.24vw;border-radius:50%;z-index:1}
.orbit::after{content:'';position:absolute;left:50%;top:-0.6vw;width:1vw;height:1vw;border-radius:50%;transform:translateX(-50%)}
.orbit.o1{width:13vw;height:13vw;margin:-6.5vw 0 0 -6.5vw;animation:spin 18s linear infinite}
.orbit.o1::after{background:radial-gradient(circle,#fff,#ffd36b);box-shadow:0 0 14px 3px rgba(255,211,107,.95)}
.orbit.o2{width:15vw;height:15vw;margin:-7.5vw 0 0 -7.5vw;animation:spin 26s linear infinite reverse}
.orbit.o2::after{background:radial-gradient(circle,#fff,#5fc89a);box-shadow:0 0 14px 3px rgba(95,200,154,.95)}
.orbit.o3{width:17vw;height:17vw;margin:-8.5vw 0 0 -8.5vw;animation:spin 34s linear infinite}
.orbit.o3::after{width:0.76vw;height:0.76vw;background:radial-gradient(circle,#fff,#9fe3c0);box-shadow:0 0 12px 3px rgba(159,227,192,.9)}
.welcome .wtitle{font-size:18px;font-weight:650;color:#ffe6b0;text-shadow:0 1px 6px rgba(0,0,0,.6)}
.welcome .wsub{font-size:13.5px;color:rgba(238,246,240,.92);max-width:460px;line-height:1.55;text-shadow:0 1px 5px rgba(0,0,0,.55)}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes aurapulse{0%,100%{opacity:.65;transform:scale(1)}50%{opacity:1;transform:scale(1.06)}}
@keyframes advance{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-9px) scale(1.035)}}
@media (prefers-reduced-motion:reduce){.orbit,.orbsys .ava,.orbsys .aura{animation:none}}
.msg{display:flex;flex-direction:column;gap:8px}
.role{font-family:var(--mono);font-size:11px;text-transform:uppercase;color:rgba(233,242,237,.78);text-shadow:0 1px 4px rgba(0,0,0,.5)}
.bubble{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;white-space:pre-wrap;font-size:15px}
.msg.user .bubble{background:var(--surface-2)}.msg.assistant .bubble{border-left:3px solid var(--petrol)}
.msg.emergency .bubble{border-left:3px solid var(--red);background:var(--red-bg)}
.msg.refusal .bubble{border-left:3px solid var(--amber);background:var(--amber-bg)}
.trace{display:flex;gap:6px;flex-wrap:wrap;font-family:var(--mono);font-size:11px}
.step{border:1px solid var(--line);border-radius:6px;padding:2px 8px;color:var(--muted);background:var(--surface)}
.step.active{color:var(--petrol-deep);border-color:var(--petrol);background:#e8f2f0}.step.done{color:var(--ink)}
.step .ms{opacity:.6}
.sources{display:flex;flex-direction:column;gap:6px}.sources .head{font-family:var(--mono);font-size:11px;color:var(--muted)}
.cite{border:1px solid var(--line);border-radius:8px;padding:8px 11px;background:var(--surface-2);font-size:13px}
.cite .tag{font-family:var(--mono);color:var(--petrol-deep);font-weight:600}
.cite .meta{color:var(--muted);font-size:12px;font-family:var(--mono);margin-top:2px}
footer{position:relative;z-index:2;background:transparent;padding:14px 20px}
.composer{max-width:780px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;resize:none;border:1px solid rgba(255,255,255,.45);border-radius:10px;padding:11px 13px;font-size:15px;background:rgba(255,255,255,.86);min-height:46px}
textarea:focus{outline:2px solid var(--petrol)}
.mic{width:46px;height:46px;border-radius:10px;border:1px solid rgba(255,255,255,.45);background:rgba(255,255,255,.86);font-size:18px}
.mic.listening{background:var(--red-bg);border-color:var(--red)}
.send{background:var(--petrol);color:#fff;border:0;border-radius:10px;padding:0 20px;height:46px;font-size:15px;font-weight:600}
.send:disabled{background:var(--muted)}
.vhint{max-width:780px;margin:6px auto 0;font-size:11px;color:rgba(232,242,237,.7);font-family:var(--mono);text-shadow:0 1px 4px rgba(0,0,0,.6)}
</style></head><body>
<div class="stage"><div class="orbsys"><div class="aura"></div><div class="ava" role="img" aria-label="Lexa"></div><span class="orbit o3"></span><span class="orbit o2"></span><span class="orbit o1"></span><div class="lextitle">I'm Lexa, your grounded medical assistant</div></div></div>
<header><div class="bar"><span class="logo"><img src="/assets/logo.png" alt=""></span><h1>LEXA</h1>
<span class="sub">Lab EXplanation Assistant</span><span class="spacer"></span>
<span class="datelbl" id="datelbl"></span><span class="clock" id="clock"></span>
<button class="ibtn" id="speak">&#128264; Speak: off</button>
<button class="ibtn" id="wake">&#127897;&#65039; Lexa: off</button>
<a class="navlink" href="/dashboard">Dashboard</a><span class="who" id="who"></span>
<button class="ibtn" id="logout">Sign out</button></div>
<div class="disc">Informational decision support for healthcare professionals - not a medical device, not a diagnosis. In an emergency, contact emergency services.</div>
</header>
<main><div class="feed" id="feed"><div class="empty" id="empty"></div></div></main>
<footer><div class="composer">
<textarea id="input" placeholder="Ask a clinical reference question..." rows="1"></textarea>
<button class="mic" id="mic" title="Dictate">&#127908;</button>
<button class="send" id="send">Ask</button></div>
<div class="vhint" id="vhint"></div></footer>
<script>
const token=localStorage.getItem("ma_token");if(!token)location.replace("/login");
const STEPS=["input_guard","retrieval","synthesis","output_guard"];
const LBL={input_guard:"safety in",retrieval:"retrieval",synthesis:"synthesis",output_guard:"safety out"};
const feed=document.getElementById("feed"),input=document.getElementById("input"),send=document.getElementById("send");
const micBtn=document.getElementById("mic"),speakBtn=document.getElementById("speak"),wakeBtn=document.getElementById("wake");
let ws,current=null,speakOn=false;
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e;}
function logout(){localStorage.clear();location.replace("/login");}
document.getElementById("logout").onclick=logout;
(function clock(){const d=new Date();const c=document.getElementById("clock");if(c)c.textContent=String(d.getHours()).padStart(2,"0")+":"+String(d.getMinutes()).padStart(2,"0");
 const dl=document.getElementById("datelbl");if(dl){const dn=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][d.getDay()];const mn=["January","February","March","April","May","June","July","August","September","October","November","December"][d.getMonth()];const day=d.getDate();const o=(day%10==1&&day!=11)?"st":(day%10==2&&day!=12)?"nd":(day%10==3&&day!=13)?"rd":"th";dl.textContent=dn+" "+day+o+" "+mn+" "+d.getFullYear()+"  ";}
 setTimeout(clock,1000);})();
fetch("/v1/me",{headers:{authorization:"Bearer "+token}}).then(r=>{if(!r.ok)throw 0;return r.json();})
 .then(me=>document.getElementById("who").textContent=me.display_name||me.username).catch(logout);
function connect(){const p=location.protocol==="https:"?"wss":"ws";
 ws=new WebSocket(p+"://"+location.host+"/v1/ws/chat?token="+encodeURIComponent(token));
 ws.onmessage=ev=>handle(JSON.parse(ev.data));
 ws.onclose=e=>{send.disabled=true;if(e.code===4401){logout();return;}setTimeout(connect,1500);};
 ws.onopen=()=>{send.disabled=false;};}
connect();
function newTrace(){const w=el("div","trace"),s={};STEPS.forEach(x=>{const d=el("div","step",LBL[x]);s[x]=d;w.appendChild(d);});return{w,s};}
function startA(){document.getElementById("empty")?.remove();const m=el("div","msg assistant");
 m.appendChild(el("div","role","LEXA"));const tr=newTrace();m.appendChild(tr.w);
 const b=el("div","bubble","...");m.appendChild(b);const so=el("div","sources");so.style.display="none";m.appendChild(so);
 feed.appendChild(m);scroll();return{m,tr,b,so};}
function handle(d){if(d.type==="error"){if(current)current.b.textContent="Error: "+d.message;return;}if(!current)return;
 if(d.type==="stage"){const s=current.tr.s[d.name];if(s){STEPS.forEach(x=>current.tr.s[x].classList.remove("active"));s.classList.add("active");}}
 if(d.type==="final"){Object.values(current.tr.s).forEach(s=>s.classList.remove("active"));
  for(const[s,ms]of Object.entries(d.timings_ms||{})){const st=current.tr.s[s];if(st){st.classList.add("done");st.appendChild(el("span","ms"," "+ms+"ms"));}}
  current.b.textContent=d.answer;if(d.action==="emergency")current.m.className="msg emergency";else if(d.refused)current.m.className="msg refusal";
  renderSrc(current.so,d.citations||[]);if(speakOn)speak(d.answer);current=null;send.disabled=false;scroll();}}
function renderSrc(box,c){box.innerHTML="";if(!c.length){box.style.display="none";return;}box.style.display="flex";
 box.appendChild(el("div","head","SOURCES ("+c.length+")"));c.forEach(x=>{const card=el("div","cite");
 card.appendChild(el("span","tag","["+x.label+"] "));card.appendChild(document.createTextNode(x.doc_title));
 card.appendChild(el("div","meta",x.section+" \u00b7 v"+x.version+" \u00b7 "+x.chunk_id));box.appendChild(card);});}
function ask(){const q=input.value.trim();if(!q||!ws||ws.readyState!==1)return;document.getElementById("empty")?.remove();
 const um=el("div","msg user");um.appendChild(el("div","role","You"));um.appendChild(el("div","bubble",q));feed.appendChild(um);
 current=startA();send.disabled=true;ws.send(JSON.stringify({query:q}));input.value="";autosize();scroll();}
function scroll(){const m=document.querySelector("main");m.scrollTop=m.scrollHeight;}
function autosize(){input.style.height="auto";input.style.height=Math.min(input.scrollHeight,160)+"px";}
send.onclick=ask;input.addEventListener("input",autosize);
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask();}});
const synth=window.speechSynthesis;
let lexaVoice=null;
function pickVoice(){if(!synth)return;const vs=synth.getVoices();if(!vs.length)return;
 const en=vs.filter(v=>/en[-_]?us/i.test(v.lang));
 const pref=["Aria","Jenny","Michelle","Ava","Samantha","Google US English","Zira"];
 for(const n of pref){const m=(en.length?en:vs).find(v=>v.name.toLowerCase().includes(n.toLowerCase()));if(m){lexaVoice=m;return;}}
 lexaVoice=en.find(v=>/female|aria|jenny|zira|samantha|michelle|ava|eva/i.test(v.name))||en[0]||vs[0];}
if(synth){pickVoice();synth.addEventListener("voiceschanged",pickVoice);}
function utter(t){const u=new SpeechSynthesisUtterance(t);u.lang="en-US";if(lexaVoice)u.voice=lexaVoice;u.rate=1.3;u.pitch=0.82;return u;}
function speak(t){if(!synth)return;synth.cancel();synth.speak(utter(t));}
if(!synth)speakBtn.style.display="none";
speakBtn.onclick=()=>{speakOn=!speakOn;speakBtn.classList.toggle("on",speakOn);
 speakBtn.textContent=speakOn?"\u{1F50A} Speak: on":"\u{1F508} Speak: off";if(!speakOn&&synth)synth.cancel();};
// ---- offline voice: mic capture (Web Audio) + streaming STT (Vosk over WebSocket) ----
const vhint=document.getElementById("vhint");
const LISTEN_MSG="\u{1F399}️ Listening… just say “Lexa” to wake her — no buttons needed. Speech is processed offline on your device.";
const HAVE_MIC=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)&&!!window.WebSocket&&!!(window.AudioContext||window.webkitAudioContext);
let voiceWS=null,voiceReady=false,sending=false,awake=false,wakeOn=true;
let audioCtx=null,micStream=null,srcNode=null,procNode=null;
function vsend(o){try{if(voiceWS&&voiceWS.readyState===1)voiceWS.send(JSON.stringify(o));}catch(_){}}
function resetRec(){vsend({cmd:"reset"});}
// “Lexa” is out-of-vocabulary in the compact offline model, which hears it as
// “lexa/lex/lexus/like so/look so/leg so/alexa…”. Match those tolerantly so the
// wake word works hands-free; the mic button stays a 100%-reliable trigger.
const WAKE_RE=/(?:^|\b)(?:hey |ok |okay )?(?:a?lex(?:a|i|ie|y|us|ar)?|lex a|leksa|like so|look so|leg so|lik so|leg sa)\b/i;
function afterWake(t){const mm=(t||"").match(WAKE_RE);return mm?t.slice(mm.index+mm[0].length).replace(/^[\s,.;:!?-]+/,"").trim():"";}
function stripWake(t){const mm=(t||"").match(WAKE_RE);return mm?(t.slice(0,mm.index)+" "+t.slice(mm.index+mm[0].length)).replace(/\s+/g," ").trim():(t||"").trim();}
function setGlow(on){document.querySelector(".orbsys")?.classList.toggle("awake",on);micBtn.classList.toggle("listening",on);}
function enableSpeak(){if(!speakOn){speakOn=true;speakBtn.classList.add("on");speakBtn.textContent="\u{1F50A} Speak: on";}}
let wakeTimer=null;
function wakeGlow(greet){awake=true;setGlow(true);enableSpeak();
 if(greet){const g=utter("Yes, I'm listening");synth&&synth.cancel();synth&&synth.speak(g);}
 vhint.textContent="\u{1F399}️ Yes? I'm listening — ask your question.";
 clearTimeout(wakeTimer);wakeTimer=setTimeout(()=>{if(awake)sleepLexa();},14000);}
function sleepLexa(){awake=false;setGlow(false);clearTimeout(wakeTimer);vhint.textContent=LISTEN_MSG;resetRec();}
function askVoice(q){q=(q||"").trim();if(!q)return;clearTimeout(wakeTimer);input.value=q;autosize();ask();
 awake=false;setGlow(false);setTimeout(()=>{vhint.textContent=LISTEN_MSG;resetRec();},600);}
function onVoiceText(text,isFinal){text=(text||"").trim();if(!text||!wakeOn)return;
 if(awake){const q=stripWake(text);            // ignore the wake-word echo; wait for the real question
  if(!q){input.value="";return;}
  input.value=q;autosize();if(isFinal)askVoice(q);return;}
 const wm=WAKE_RE.test(text);
 const shortCall=isFinal&&text.split(/\s+/).filter(Boolean).length<=2; // a brief utterance while idle = her name
 if(wm||shortCall){const after=wm?afterWake(text):"";
  if(after){wakeGlow(false);resetRec();askVoice(after);}  // "Lexa, <question>" in one breath
  else{wakeGlow(true);resetRec();}}}                       // just her name -> wake + greet, then listen
function downsample(buf,inR,outR){if(outR>=inR)return buf;const ratio=inR/outR,n=Math.floor(buf.length/ratio),out=new Float32Array(n);
 for(let o=0;o<n;o++){const s=Math.floor(o*ratio),e=Math.floor((o+1)*ratio);let a=0,c=0;for(let j=s;j<e&&j<buf.length;j++){a+=buf[j];c++;}out[o]=c?a/c:(buf[s]||0);}return out;}
function floatTo16(buf){const out=new Int16Array(buf.length);for(let i=0;i<buf.length;i++){let s=Math.max(-1,Math.min(1,buf[i]));out[i]=s<0?s*0x8000:s*0x7fff;}return out;}
async function startAudio(){if(audioCtx)return;
 try{micStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});}
 catch(e){vhint.textContent="Microphone blocked — allow mic access to use voice.";return;}
 audioCtx=new (window.AudioContext||window.webkitAudioContext)();try{await audioCtx.resume();}catch(_){}
 srcNode=audioCtx.createMediaStreamSource(micStream);procNode=audioCtx.createScriptProcessor(4096,1,1);
 srcNode.connect(procNode);procNode.connect(audioCtx.destination);
 procNode.onaudioprocess=ev=>{if(!sending)return;if(synth&&synth.speaking)return;if(!voiceWS||voiceWS.readyState!==1)return;
  const inB=ev.inputBuffer.getChannelData(0);voiceWS.send(floatTo16(downsample(inB,audioCtx.sampleRate,16000)).buffer);};}
function connectVoice(){if(!HAVE_MIC){if(micBtn)micBtn.disabled=true;if(wakeBtn)wakeBtn.disabled=true;vhint.textContent="Voice isn't supported in this browser.";return;}
 if(voiceWS&&(voiceWS.readyState===0||voiceWS.readyState===1))return;
 const p=location.protocol==="https:"?"wss":"ws";voiceWS=new WebSocket(p+"://"+location.host+"/v1/ws/voice?token="+encodeURIComponent(token));voiceWS.binaryType="arraybuffer";
 voiceWS.onmessage=ev=>{let d;try{d=JSON.parse(ev.data);}catch(_){return;}
  if(d.type==="ready"){voiceReady=true;startAudio();if(wakeOn)sending=true;}
  else if(d.type==="unavailable"){vhint.textContent="Offline voice model not found — reinstall the app to enable voice.";if(wakeBtn)wakeBtn.disabled=true;if(micBtn)micBtn.disabled=true;}
  else if(d.type==="partial")onVoiceText(d.text,false);
  else if(d.type==="final")onVoiceText(d.text,true);};
 voiceWS.onclose=()=>{voiceReady=false;if(wakeOn)setTimeout(connectVoice,1500);};
 voiceWS.onerror=()=>{};}
if(!synth){}/* speak button handled above */
micBtn.onclick=()=>{if(!HAVE_MIC)return;if(!voiceReady){connectVoice();}sending=true;wakeGlow(false);resetRec();};
if(wakeBtn){wakeBtn.onclick=()=>{wakeOn=!wakeOn;wakeBtn.classList.toggle("on",wakeOn);
 wakeBtn.textContent=wakeOn?"\u{1F399}️ Lexa: on":"\u{1F399}️ Lexa: off";
 if(wakeOn){sending=true;connectVoice();vhint.textContent=LISTEN_MSG;}
 else{sending=false;awake=false;setGlow(false);vhint.textContent="Voice paused — toggle the mic button to resume.";}};}
// hands-free: open the offline voice channel on load — just say “Lexa”, no button press
wakeOn=true;if(wakeBtn){wakeBtn.classList.add("on");wakeBtn.textContent="\u{1F399}️ Lexa: on";}
if(HAVE_MIC){vhint.textContent=LISTEN_MSG;connectVoice();
 const resume=()=>{try{audioCtx&&audioCtx.resume();}catch(_){}};
 document.addEventListener("click",resume,{once:true});
 document.addEventListener("visibilitychange",()=>{if(!document.hidden&&wakeOn)connectVoice();});
 window.addEventListener("focus",()=>{if(wakeOn)connectVoice();});}
else{vhint.textContent="Voice isn't supported in this browser.";if(wakeBtn)wakeBtn.disabled=true;if(micBtn)micBtn.disabled=true;}
</script></body></html>""".replace("__CSS__", _CSS).replace("__CHATBG__", _page_bg_css())

DASH_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LEXA - dashboard</title>
<style>
*{margin:0;box-sizing:border-box}html,body{height:100%}
body{background:#000;display:flex;align-items:center;justify-content:center;overflow:auto}
img.dash{max-width:100%;max-height:100vh;display:block;cursor:pointer}
.back{position:fixed;top:14px;left:16px;color:#ffe6b0;font-family:system-ui,-apple-system,sans-serif;font-size:13px;font-weight:600;text-decoration:none;text-shadow:0 1px 6px #000;z-index:2}
</style></head><body>
<a class="back" href="/">&#8592; Chat</a>
<img class="dash" src="/assets/dash_page.jpg" alt="LEXA dashboard" onclick="location.href='/'">
<script>if(!localStorage.getItem("ma_token"))location.replace("/login");</script>
</body></html>"""

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="MedAssist", version="1.0-single")


def require_user(authorization: str | None = Header(default=None)) -> dict:
    if not AUTH_ENABLED:
        return {"sub": "local", "name": "Local", "role": "admin"}
    claims = verify_token((authorization or "").removeprefix("Bearer ").strip())
    if not claims:
        raise HTTPException(status_code=401, detail="not authenticated")
    return claims


class LoginReq(BaseModel):
    username: str
    password: str


class ChatReq(BaseModel):
    query: str


@app.get("/")
def page_index():
    return HTMLResponse(CHAT_HTML)


@app.get("/login")
def page_login():
    return HTMLResponse(LOGIN_HTML)


@app.get("/dashboard")
def page_dashboard():
    return HTMLResponse(DASH_HTML)


@app.get("/assets/{name}")
def asset(name: str):
    if name not in {"avatar.png", "dash_bg.jpg", "dash_page.jpg", "logo.png"}:
        raise HTTPException(status_code=404, detail="not found")
    path = _asset_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


@app.post("/v1/auth/login")
def login(req: LoginReq):
    if not AUTH_ENABLED:
        u = {"username": "local", "display_name": "Local", "role": "admin"}
        return JSONResponse({"token": issue_token(u), "user": u})
    user = verify_login(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid username or password")
    return JSONResponse({"token": issue_token(user), "user": user})


@app.get("/v1/me")
def me(user: dict = Depends(require_user)):
    return JSONResponse({"username": user["sub"], "display_name": user.get("name"),
                         "role": user.get("role")})


@app.get("/v1/health")
def health():
    return JSONResponse({"status": "ok", "provider": PROVIDER, "model": MODEL_NAME,
                         "indexed_chunks": len(INDEX.chunks), "auth_enabled": AUTH_ENABLED,
                         "chat_llm": CHAT_LLM, "chat_model": CHAT_MODEL,
                         "chat_llm_online": _ollama_available() if CHAT_LLM else False})


@app.get("/v1/stats")
def stats(user: dict = Depends(require_user)):
    s = audit_summary()
    s.update({"indexed_chunks": len(INDEX.chunks), "provider": PROVIDER, "model": MODEL_NAME})
    return JSONResponse(s)


@app.get("/v1/audit")
def audit_feed(limit: int = 50, user: dict = Depends(require_user)):
    rows = [{"trace_id": e["trace_id"], "ts": e["ts"], "user": e.get("user"),
             "query": e["query"], "action": e.get("action"), "refused": e.get("refused"),
             "top": (e.get("retrieved") or [{}])[0].get("score") if e.get("retrieved") else None}
            for e in audit_read(limit)]
    return JSONResponse({"events": rows})


@app.post("/v1/chat")
async def chat(req: ChatReq, user: dict = Depends(require_user)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    loop = asyncio.get_running_loop()
    return JSONResponse(await loop.run_in_executor(None, answer, req.query, user["sub"], None))


@app.websocket("/v1/ws/chat")
async def ws_chat(ws: WebSocket):
    username = "local"
    if AUTH_ENABLED:
        claims = verify_token(ws.query_params.get("token", ""))
        if not claims:
            await ws.close(code=4401)
            return
        username = claims["sub"]
    await ws.accept()
    loop = asyncio.get_running_loop()
    convo: list = []  # rolling multi-turn memory for the conversation LLM
    try:
        while True:
            msg = await ws.receive_json()
            query = (msg or {}).get("query", "").strip()
            if not query:
                await ws.send_json({"type": "error", "message": "empty query"})
                continue
            queue: asyncio.Queue = asyncio.Queue()

            def emit(kind, data):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": kind, **data})

            hist = list(convo)
            task = loop.run_in_executor(None, lambda: answer(query, username, emit, hist))
            while True:
                try:
                    await ws.send_json(await asyncio.wait_for(queue.get(), timeout=0.05))
                except asyncio.TimeoutError:
                    pass
                if task.done() and queue.empty():
                    break
            result = await task
            await ws.send_json({"type": "final", **result})
            convo.append({"role": "user", "content": query})
            convo.append({"role": "assistant", "content": result.get("answer", "")})
            del convo[:-12]
    except WebSocketDisconnect:
        return


@app.get("/v1/voice/status")
def voice_status(user: dict = Depends(require_user)):
    """Tell the UI whether offline speech-to-text is available."""
    return JSONResponse({"available": os.path.isdir(_vosk_model_path())})


@app.websocket("/v1/ws/voice")
async def ws_voice(ws: WebSocket):
    """Offline speech-to-text: browser streams 16 kHz mono PCM16, we stream text back."""
    if AUTH_ENABLED:
        claims = verify_token(ws.query_params.get("token", ""))
        if not claims:
            await ws.close(code=4401)
            return
    await ws.accept()
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, _get_vosk_model)
    if model is None:
        await ws.send_json({"type": "unavailable",
                            "message": "offline voice model not available"})
        await ws.close()
        return
    import vosk
    rec = vosk.KaldiRecognizer(model, 16000)
    last_partial = ""
    await ws.send_json({"type": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                txt = msg.get("text")
                if txt:
                    try:
                        ctrl = json.loads(txt)
                    except Exception:
                        ctrl = {}
                    if ctrl.get("cmd") == "reset":
                        rec = vosk.KaldiRecognizer(model, 16000)
                        last_partial = ""
                continue
            if await loop.run_in_executor(None, rec.AcceptWaveform, data):
                text = json.loads(rec.Result()).get("text", "").strip()
                last_partial = ""
                if text:
                    await ws.send_json({"type": "final", "text": text})
            else:
                part = json.loads(rec.PartialResult()).get("partial", "").strip()
                if part and part != last_partial:
                    last_partial = part
                    await ws.send_json({"type": "partial", "text": part})
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


def _serve():
    import uvicorn
    # Pin pure-Python protocol impls so the packaged (.exe) build bundles cleanly.
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning",
                loop="asyncio", http="h11", ws="websockets")


def run_desktop():
    """Native-window mode for the packaged desktop build (.exe).

    Serves the app on a daemon thread and shows the UI in a native window via
    pywebview - an OPTIONAL extra needed only for the desktop/exe build; the
    core app still runs on fastapi+uvicorn+numpy alone. Closing the window ends
    the process.
    """
    import os
    import sys
    import threading
    import time
    import urllib.error
    # frozen windowed builds have no console - keep stdout/stderr writable
    for _stream in ("stdout", "stderr"):
        if getattr(sys, _stream) is None:
            setattr(sys, _stream, open(os.devnull, "w"))
    # let WebView2 use the microphone without a permission prompt (for voice / wake word)
    os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
                          "--use-fake-ui-for-media-stream")
    import webview  # optional: pip install pywebview
    threading.Thread(target=_serve, daemon=True).start()
    base = f"http://{HOST}:{PORT}"
    for _ in range(90):  # wait up to ~30s for the server to answer (be tolerant of slow first call)
        try:
            urllib.request.urlopen(base + "/v1/health", timeout=2)
            break
        except Exception:  # URLError, TimeoutError/socket.timeout, OSError - just keep waiting
            time.sleep(0.3)
    webview.create_window("LEXA - Lab EXplanation Assistant", base + "/login",
                          width=1180, height=820, min_size=(900, 640))
    webview.start()


if __name__ == "__main__":
    import sys
    if getattr(sys, "frozen", False) or "--desktop" in sys.argv:
        run_desktop()
    else:
        print(f"\n  MedAssist -> http://{HOST}:{PORT}   (login: {ADMIN_USERNAME})")
        print(f"  provider={PROVIDER}  model={MODEL_NAME}  chunks={len(INDEX.chunks)}  "
              f"auth={'on' if AUTH_ENABLED else 'off'}\n")
        _serve()
