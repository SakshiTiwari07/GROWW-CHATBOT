"""Ask API. Embeddings stay on the Hugging Face model. The reply is written by Groq."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="The page file is missing from this deployment.")
    return FileResponse(page)


@app.post("/ask", response_model=AskBody)
def ask_route(body: AskRequest) -> AskBody:
    from app.ask.service import ask
    from app.config import load_settings

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
