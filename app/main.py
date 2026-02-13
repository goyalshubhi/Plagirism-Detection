import os
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.v1.routes import router as v1_router
# from app.api.admin import router as admin_router

app = FastAPI(title="Plagiarism Detection API")

allowed_origins = ["http://localhost:5173", "http://localhost:80"]
if os.getenv("ENVIRONMENT") == "development":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router, prefix="/api/auth")
# app.include_router(admin_router, prefix="/api/admin")
app.include_router(v1_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Plagiarism Detection API", "status": "running"}
