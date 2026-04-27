import os
import json
import httpx
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import MaterialMedicaRAG

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
PDF_PATHS     = [
    BASE_DIR / "data" / "robin_murphy_searchable.pdf",
]
DB_PATH       = str(BASE_DIR / "data" / "chroma_db")
FRONT_DIR     = str(BASE_DIR / "frontend")
MEMORY_FILE   = BASE_DIR / "data" / "conversations.json"
OLLAMA_URL    = "http://localhost:11434/api/chat"
MODEL         = "llama3"
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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
- Keep responses focused and practical — the client wants guidance, not a lecture
- Never cite sources, never say where information comes from, never mention book names — speak naturally as a knowledgeable practitioner
- NEVER use the words "Material Medica", "Materia Medica", or any book title in your response under any circumstances

DISCLAIMER: Always include a gentle reminder that recommendations are for educational 
purposes, complement but do not replace professional medical care, and that serious 
or urgent symptoms require immediate medical attention."""

# ── Memory ─────────────────────────────────────────────────────────────────
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

# ── RAG ────────────────────────────────────────────────────────────────────
rag = MaterialMedicaRAG([str(p) for p in PDF_PATHS], DB_PATH)

GDRIVE_FILES = {
    "robin_murphy_searchable.pdf": "1BOgl_K8b9fTa_i_oHg22_iXbPmoYufai",
}

def download_from_gdrive(file_id: str, dest_path: Path):
    import requests
    print(f"⬇️  Downloading {dest_path.name} from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"confirm": token}, stream=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"✅ Downloaded {dest_path.name}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download PDFs from Google Drive if not present
    for filename, file_id in GDRIVE_FILES.items():
        dest = BASE_DIR / "data" / filename
        if not dest.exists():
            download_from_gdrive(file_id, dest)

    if Path(DB_PATH).exists():
        rag.load()
    else:
        if not any(p.exists() for p in PDF_PATHS):
            print("⚠️  No PDFs found in data/ folder")
        else:
            rag.index()
    yield

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Logos AI", lifespan=lifespan)

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
    return {"status": "ok", "model": "anthropic" if USE_ANTHROPIC else MODEL}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    context = rag.search(last_user, k=4)
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\n═══ RELEVANT KNOWLEDGE ═══\n\n{context}\n\n═══════════════════════════════════════════════════"

    try:
        if USE_ANTHROPIC:
            import anthropic
            ac = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = ac.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in req.messages]
            )
            reply = response.content[0].text
        else:
            ollama_messages = [{"role": "system", "content": system}]
            ollama_messages += [{"role": m.role, "content": m.content} for m in req.messages]
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
        if USE_ANTHROPIC:
            import anthropic
            ac = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = ac.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.content[0].text
        else:
            ollama_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={"model": MODEL, "messages": ollama_messages, "stream": False},
                )
                data = response.json()
                reply = data["message"]["content"]
        return {"content": [{"text": reply}]}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve frontend (must be last) ──────────────────────────────────────────
if Path(FRONT_DIR).exists():
    app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")