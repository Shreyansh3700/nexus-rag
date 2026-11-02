import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.config.config import set_environment
from src.logger import configure_logging, get_logger
from src.rate_limit import limiter
from src.routes import auth_router, chat_router, documents_router, sessions_router

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="nexus-rag",
    description="Complete Chatbot with persistent history and RAG",
    version="0.1.0",
    lifespan=set_environment,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Allows the React SPA (served from a different origin/port than the API)
# to call these endpoints from the browser, including the Authorization header.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(documents_router)


@app.get("/")
async def index():
    logger.debug("Health check requested")
    return {"service": "nexus-rag", "status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
