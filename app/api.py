"""Ask API. Embeddings stay on the Hugging Face model. The reply is written by Groq."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ask.service import ask
from app.config import load_settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="HDFC Mutual Fund Facts")


class AskRequest(BaseModel):
    question: str = Field(default="")


class AskBody(BaseModel):
    answer: str
    citation_url: str
    last_updated: str
    refused: bool


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask", response_model=AskBody)
def ask_route(body: AskRequest) -> AskBody:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    response = ask(question, load_settings())
    return AskBody(
        answer=response.answer,
        citation_url=response.citation_url,
        last_updated=response.last_updated,
        refused=response.refused,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
