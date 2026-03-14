from fastapi import FastAPI,Request,Response
from pydantic import BaseModel
import uuid
import uvicorn
import os

from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.ragchain import rag_chain_with_memory

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_question(query: QueryRequest, request: Request, response: Response):

    session_id = request.cookies.get("session_id")

    if session_id is None:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)

    async def generate():

        for chunk in rag_chain_with_memory.stream(
            {"input": query.question},
            config={"configurable": {"session_id": session_id}}
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)