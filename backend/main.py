import os
import json
from datetime import datetime
import httpx
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import MaterialMedicaRAG

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
PDF_PATHS = [
    BASE_DIR / "data" / "The_Material_Medica_of_Narayani_Combination_Remedies.pdf",
    BASE_DIR / "data" / "robin_murphy_searchable.pdf",
]
DB_PATH   = str(BASE_DIR / "data" / "chroma_db")
FRONT_DIR = str(BASE_DIR / "frontend")
MEMORY_FILE = BASE_DIR / "data" / "conversations.json"

def save_conversation(messages: list, response: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "response": response
    }
    conversations = []
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r") as f:
            conversations = json.load(f)
    conversations.append(entry)
    with open(MEMORY_FILE, "w") as f:
        json.dump(conversations, f, indent=2)
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL     = "llama3"

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a warm, knowledgeable homeopathic consultant with deep 
knowledge of classical and combination homeopathic remedies.

When answering:
- Respond naturally and conversationally, like a trusted practitioner
- Reference specific remedy names in **bold**
- Use bullet points for symptom lists
- Use numbered lists when ranking remedy recommendations
- Ask follow-up questions about modalities (what makes it better/worse),
  time of day, emotional state, and accompanying symptoms to narrow recommendations
- When recommending combination remedies alongside classical single remedies, 
  mention both options — when referencing Narayani combination remedies specifically, 
  always credit them as "Narayani [remedy name]" to distinguish them from classical single remedies
- Never cite sources, never say where information comes from, never mention book names — speak naturally as a knowledgeable practitioner

DISCLAIMER: Always include a gentle reminder that recommendations are for educational 
purposes, complement but do not replace professional medical care, and that serious 
or urgent symptoms require immediate medical attention."""

# ── RAG ────────────────────────────────────────────────────────────────────
rag = MaterialMedicaRAG([str(p) for p in PDF_PATHS], DB_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path(DB_PATH).exists():
        rag.load()
    else:
        if not any(p.exists() for p in PDF_PATHS):
            print("⚠️  No PDFs found in data/ folder")
            print("   Place your PDFs in the data/ folder and restart.")
        else:
            rag.index()
    yield

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Materia Medica AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ─────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class CoughRequest(BaseModel):
    duration: float
    rms: float
    bursts: int
    centroid: float
    traits: list[str]

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    context = rag.search(last_user, k=4)

    system = SYSTEM_PROMPT
    if context:
       system += f"\n\n═══ RELEVANT KNOWLEDGE ═══\n\n{context}\n\n═══════════════════════════════════════════════════"

    ollama_messages = [{"role": "system", "content": system}]
    ollama_messages += [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": MODEL, "messages": ollama_messages, "stream": False},
            )
            data = response.json()
            reply = data["message"]["content"]
            save_conversation(req.messages, reply)
            return {"content": [{"text": reply}]}
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-cough")
async def analyze_cough(req: CoughRequest):
    context = rag.search("cough remedy treatment", k=5)
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\n═══ RELEVANT KNOWLEDGE ═══\n\n{context}\n\n═══════════════════════════════════════════════════"

    prompt = f"""Cough audio analysis:
- Duration: {req.duration:.1f}s
- Intensity (RMS): {req.rms:.3f}
- Cough bursts: {req.bursts}
- Spectral centroid: {req.centroid:.0f} Hz
- Traits: {', '.join(req.traits)}

Based on these acoustic characteristics:
1. What type of cough does this suggest?
2. Top 3 homeopathic remedies that best match
3. Two follow-up questions to refine the recommendation"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            data = response.json()
            return {"content": [{"text": data["message"]["content"]}]}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # ── Serve frontend (must be last) ──────────────────────────────────────────
if Path(FRONT_DIR).exists():
    app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")