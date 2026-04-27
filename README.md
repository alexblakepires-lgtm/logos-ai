# Logos 🌿
### AI-Powered Homeopathic Consultant

A locally-running AI assistant that answers homeopathic remedy questions 
based on classical Material Medica books, with cough sound analysis.

## Features
- 🌿 RAG pipeline over classical homeopathy books
- 🎙 Cough sound recorder and analyzer
- 🤖 Runs completely locally via Ollama + Mistral
- 💾 Saves all conversations automatically
- 📖 Cites knowledge from Narayani Combination Remedies 
  and Lotus Materia Medica (Robin Murphy)

## Tech Stack
- **Backend:** Python, FastAPI, LangChain, ChromaDB
- **AI Model:** Mistral 7B via Ollama (runs locally)
- **Frontend:** HTML, CSS, React (via CDN)
- **Audio:** Web Audio API

## Setup
1. Install [Ollama](https://ollama.com) and pull Mistral:
   \`\`\`bash
   ollama pull mistral
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r backend/requirements.txt
   \`\`\`
3. Add your Material Medica PDFs to the `data/` folder
4. Start the server:
   \`\`\`bash
   cd backend
   uvicorn main:app --reload
   \`\`\`
5. Open `http://localhost:8000`

## Project Structure
\`\`\`
logos-ai/
├── backend/
│   ├── main.py      ← FastAPI server + RAG pipeline
│   └── rag.py       ← PDF indexing + vector search
├── frontend/
│   └── index.html   ← UI
└── data/            ← PDFs + ChromaDB (not included in repo)
\`\`\`

## Disclaimer
For educational purposes only. Not a substitute for professional medical care.