import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.catalogue import EntityCatalogue
from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Super Memory API")

# Enable CORS for the Next.js frontend (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

engine = MeTTaEngine()
pm = PersistenceManager(data_dir)
pm.load_state(engine)

catalogue = EntityCatalogue()
catalogue.sync(engine.get_all_atoms())

agent = NeuroSymbolicAgent(engine, catalogue, pm)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "Super Memory Neuro-Symbolic API is running"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        messages = request.history
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            
        reply = agent.chat(messages, request.message)
        return {"reply": reply}
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph")
def get_graph():
    atoms = engine.get_all_atoms()
    rules = getattr(engine, "_rules", [])
    return {
        "atoms": atoms,
        "rules": rules
    }

@app.get("/api/ledger")
def get_ledger(limit: int = 50):
    log = pm.get_audit_log(limit)
    return {"log": log}

@app.post("/api/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests plain text files (.txt, .md, .csv, .log, .json), extracts readable content,
    and passes it into the Neuro-Symbolic engine to ground factual relations into AtomSpace.
    """
    allowed_extensions = {".txt", ".md", ".csv", ".log", ".json"}
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format '{suffix}'. Please upload text-readable files only: {', '.join(allowed_extensions)}"
        )

    try:
        contents = await file.read()
        try:
            full_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            full_text = contents.decode("latin-1", errors="ignore")
            
        full_text = full_text.strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="The uploaded text file is empty.")

        # Ingest into Super Memory via NeuroSymbolicAgent
        instruction = (
            f"Ingest the following text extracted from uploaded file '{file.filename}'. "
            f"Extract all factual statements, relationships, entities, and attributes, and store them "
            f"into the knowledge graph using learn_fact:\n\n{full_text[:4000]}"
        )
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        reply = agent.chat(messages, instruction)
        
        # Sync catalogue with fresh atoms
        current_atoms = engine.get_all_atoms()
        catalogue.sync(current_atoms)
        
        return {
            "status": "success",
            "filename": file.filename,
            "characters_extracted": len(full_text),
            "summary": reply,
            "total_atoms": len(current_atoms)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
