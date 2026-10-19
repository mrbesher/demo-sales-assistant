import json
import os
import threading
import uuid
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from calculations import ProcessData
from conversation import Conversation
from evidence import Evidence
from model import Model
from retrieval import Encoder, HybridIndex

base = Path(__file__)
base = base.parent
model = Model()
storage_name = os.getenv("STORAGE_DIR", "storage")
storage = Path(storage_name)
if not storage.is_absolute():
    storage = base / storage
manifest_path = storage / "manifest.json"
service = None
sessions = {}
load_lock = threading.Lock()
sessions_lock = threading.Lock()
app = FastAPI(title="Metso staff assistant")


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=64)


@app.get("/health")
def health():
    prepared = manifest_path.exists()
    configured = bool(model.token)
    ready = prepared and configured
    message = "Ready."
    if not prepared:
        message = "Prepare the supplied data with prepare.py before starting a conversation."
    elif not configured:
        message = "Set HF_TOKEN in .env, then restart the server."
    return {"ready": ready, "model_configured": configured, "message": message}


@app.post("/chat")
def chat(request: ChatInput):
    global service
    question = request.message.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")
    if not model.token:
        raise HTTPException(503, "Set HF_TOKEN in .env and restart the server.")
    with load_lock:
        if service is None:
            if not manifest_path.exists():
                raise HTTPException(503, "Run prepare.py with the supplied data first.")
            try:
                manifest_text = manifest_path.read_text(encoding="utf-8")
                manifest = json.loads(manifest_text)
                encoder = Encoder(
                    manifest["embedding_model"], revision=manifest["embedding_revision"]
                )
                chunk_path = storage / "chunks.json"
                report_path = storage / "reports.json"
                chunk_text = chunk_path.read_text(encoding="utf-8")
                report_text = report_path.read_text(encoding="utf-8")
                chunks = json.loads(chunk_text)
                reports = json.loads(report_text)
                chunk_vectors = np.load(storage / "chunks.npy", allow_pickle=False)
                report_vectors = np.load(storage / "reports.npy", allow_pickle=False)
                brochure_index = HybridIndex(chunks, chunk_vectors, encoder)
                report_index = HybridIndex(reports, report_vectors, encoder)
                process = ProcessData(manifest["process_file"])
                service = Evidence(brochure_index, report_index, process, storage, model)
            except (ImportError, OSError, ValueError, KeyError) as error:
                raise HTTPException(
                    503,
                    "Prepared data could not be loaded. Check the embeddings "
                    "extra and rerun prepare.py if the source files moved.",
                ) from error
    with sessions_lock:
        session_id = request.session_id
        if session_id is None:
            identifier = uuid.uuid4()
            session_id = identifier.hex
            conversation = Conversation(service, model)
            lock = threading.Lock()
            sessions[session_id] = (conversation, lock)
        elif session_id not in sessions:
            raise HTTPException(404, "This conversation has ended. Start a new conversation.")
        conversation, lock = sessions[session_id]
    acquired = lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(409, "A response is already running in this conversation.")
    try:
        result = conversation.reply(question)
    except httpx.HTTPError as error:
        if request.session_id is None:
            with sessions_lock:
                sessions.pop(session_id, None)
        raise HTTPException(
            502, "The model provider could not be reached. Please retry."
        ) from error
    except (RuntimeError, ValueError) as error:
        if request.session_id is None:
            with sessions_lock:
                sessions.pop(session_id, None)
        detail = str(error)
        raise HTTPException(502, detail) from error
    finally:
        lock.release()
    result["session_id"] = session_id
    return result


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    with sessions_lock:
        removed = sessions.pop(session_id, None)
    return {"deleted": removed is not None}


@app.get("/sources/{source_id}")
def source(source_id: str, session_id: str):
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(404, "This conversation has ended.")
    conversation = entry[0]
    record = conversation.state["sources"].get(source_id)
    if record is None:
        raise HTTPException(404, "This source has not been opened in this conversation.")
    images = []
    for page in record["pages"]:
        images.append({"page": page, "url": f"/pages/{page}?session_id={session_id}"})
    fields = record.get("fields", {})
    references = []
    for related_id in fields.get("source_ids", []):
        related = conversation.state["sources"][related_id]
        references.append({"id": related_id, "title": related["title"]})
    return {
        "id": source_id,
        "title": record["title"],
        "kind": record["kind"],
        "text": record["text"],
        "fields": fields,
        "images": images,
        "references": references,
    }


@app.get("/pages/{page_id}")
def page_image(page_id: str, session_id: str):
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(404, "This conversation has ended.")
    conversation = entry[0]
    if page_id not in conversation.state["pages"]:
        raise HTTPException(404, "This page has not been opened in this conversation.")
    for record in conversation.state["sources"].values():
        pages = zip(record["pages"], record["image_paths"], strict=True)
        for page, relative_path in pages:
            if page == page_id:
                path = storage / relative_path
                return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "The source page is unavailable.")
