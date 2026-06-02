import os
import json
import asyncio
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
    BASE_DIR / "data" / "NATURES_MATERIA_MEDICA.txt",
    BASE_DIR / "data" / "METAREPERTORY.txt",
]
DB_PATH       = str(BASE_DIR / "data" / "chroma_db")
FRONT_DIR     = str(BASE_DIR / "frontend")
MEMORY_FILE   = BASE_DIR / "data" / "conversations.json"
OLLAMA_URL    = "http://localhost:11434/api/chat"
MODEL         = "llama3"
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Logos, a precise and systematic homeopathic consultant with deep 
knowledge of classical homeopathic remedies and Material Medica.

SYMPTOM INTAKE - Always gather the complete picture systematically:
- Location (specific, one-sided, wandering?)
- Sensation (quality of pain/discomfort)
- Modalities (better/worse from: time, temperature, motion, position, pressure, weather)
- Concomitants (what else happens alongside?)
- Causation (what triggered it?)
- Mental/Emotional state during the complaint

COMMUNICATION STYLE:
- Respond naturally and conversationally, like a trusted experienced practitioner
- Be direct — tell the client what fits and what doesn't, clearly
- Never show your internal ranking process — just give clear conclusions
- Reference specific remedy names in **bold**
- Use bullet points for symptom lists
- Use numbered lists when ranking remedy recommendations
- Ask focused follow-up questions about modalities to narrow recommendations
- Keep responses focused and practical — the client wants guidance, not a lecture
- Never cite sources, never mention book names — speak naturally as a knowledgeable practitioner
- NEVER use the words "Material Medica" or any book title in your response
- Detect the language the user is writing in and always respond in that same language
- If the user writes in Portuguese, respond fully in Portuguese
- IDENTITY: You are ONLY Logos. If anyone asks who created you, who made you, or what AI you are: you MUST say "I am Logos, created by Satya11 LLC and developed by Alexandre Pires and Lua Maia, CIH. " NEVER mention Anthropic, Claude, or any underlying AI model under any circumstances.
- Never use the word "pellets" — always use "pillules" instead.

REPERTORIZATION DISCIPLINE:
- List symptoms in clear clinical language
- Compare top 3-5 remedies and highlight distinguishing features
- State clearly what confirms or rules out each remedy option
- Note what additional information would help confirm the remedy

CLINICAL PRECISION:
- When appropriate suggest potency considerations
- Mention what to expect and when to reassess
- Flag any red flags that require immediate medical attention

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
    "NATURES_MATERIA_MEDICA.txt": "16uz6fZJbTLs0AG5WP26wqGvQJyGC_7F_",
    "METAREPERTORY.txt": "1u55UWY_Cn90dhIsy2yAWTNQA1t_zCLEe",
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
    asyncio.create_task(build_database())
    yield

async def build_database():
    for filename, file_id in GDRIVE_FILES.items():
        dest = BASE_DIR / "data" / filename
        if not dest.exists():
            download_from_gdrive(file_id, dest)

    if Path(DB_PATH).exists():
        rag.load()
    else:
        chroma_zip = BASE_DIR / "data" / "chroma_db.zip"
        if not chroma_zip.exists():
            download_from_gdrive("17BMtg7Mt89FU43dvqj8x4Ca0cP-CVCxz", chroma_zip)
        import zipfile
        print("📦 Extracting ChromaDB...")
        with zipfile.ZipFile(chroma_zip, 'r') as z:
            z.extractall(BASE_DIR / "data")
        print("✅ ChromaDB extracted")
        rag.load()

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
    browser_lang: str = "en"

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
    system = SYSTEM_PROMPT + f"\n\nUSER BROWSER LANGUAGE: {req.browser_lang}. Use this as the default language unless the user writes in a different language, in which case follow what they type."
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