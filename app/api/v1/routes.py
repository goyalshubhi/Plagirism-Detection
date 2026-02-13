from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import aliased

from app.core.db import get_db
from app.models.user import User
from app.api.auth import fastapi_users
from app.core.provider_router import ProviderType

router = APIRouter()


class AnalysisOptions(BaseModel):
    provider: str = Field(default=ProviderType.LOCAL, description="AI detection provider (local, openai, together)")
    ai_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    check_plagiarism: bool = True
    check_ai: bool = True


class AnalysisResponse(BaseModel):
    batch_id: str
    status: str
    message: str


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    files: List[UploadFile] = File(default=[]),
    text: Optional[str] = Form(default=None),
    options: str = Form(default='{"provider": "local", "check_plagiarism": true, "check_ai": true}'),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user()),
):
    try:
        parsed_options = json.loads(options)
        opts = AnalysisOptions(**parsed_options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

    if not files and not text:
        raise HTTPException(status_code=400, detail="Must provide either files or text")

    from app.models import Batch, Document

    batch_id = uuid.uuid4()
    batch = Batch(
        id=batch_id,
        user_id=user.id,
        total_docs=0,
        status="queued",
        analysis_type="mixed",
        ai_provider=opts.provider,
        ai_threshold=opts.ai_threshold,
    )
    db.add(batch)

    docs_to_process = []

    # Text input
    if text:
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            batch_id=batch_id,
            filename="input_text.txt",
            storage_path=f"{batch_id}/input_text.txt",
            text_content=text,
            status="queued",
        )
        db.add(doc)
        docs_to_process.append(doc)

    # Files
    from app.services.parsing import extract_text_from_file
    from app.services.storage import StorageService
    storage_service = StorageService()

    for file in files:
        content = await file.read()
        storage_path = f"{batch_id}/{file.filename}"
        storage_service.save(storage_path, content)

        from io import BytesIO
        file_obj = BytesIO(content)
        file_obj.name = file.filename
        text_content = await extract_text_from_file(file_obj)

        doc = Document(
            batch_id=batch_id,
            filename=file.filename,
            storage_path=storage_path,
            text_content=text_content,
            status="queued",
        )
        db.add(doc)
        docs_to_process.append(doc)

    batch.total_docs = len(docs_to_process)
    await db.commit()

    # Trigger async processing
    from app.services.batch_processing import process_batch
    process_batch.delay(str(batch_id), provider=opts.provider, ai_threshold=opts.ai_threshold)

    return AnalysisResponse(batch_id=str(batch_id), status="queued", message="Analysis started successfully")


@router.get("/ai-detection/health")
async def ai_health_check():
    # instantiate inside request (no import-time heavy load)
    from app.services.ai_detection import AIDetectionService
    ai_service = AIDetectionService()
    health_status = ai_service.health_check()
    return {"service": "ai_detection", "health": health_status}


@router.post("/ai-detection")
async def detect_ai_only(
    text: str = Body(..., embed=True),
    provider: str = Body("local", embed=True),
    threshold: float = Body(0.5, embed=True),
    user: User = Depends(fastapi_users.current_user()),
):
    from app.services.ai_detection import AIDetectionService
    ai_service = AIDetectionService()

    try:
        ai_result = ai_service.detect(text, provider=provider, threshold=threshold)
        return {
            "is_ai": ai_result["is_ai"],
            "score": ai_result["score"],
            "confidence": ai_result["confidence"],
            "label": ai_result["label"],
            "provider": ai_result["provider"],
            "details": ai_result["details"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI detection failed: {str(e)}")


@router.get("/batches/{batch_id}/results")
async def get_batch_results(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user()),
):
    from app.models import Batch, Document, Comparison

    # batch ownership check
    batch_stmt = select(Batch).where(Batch.id == batch_id, Batch.user_id == user.id)
    batch_res = await db.execute(batch_stmt)
    batch = batch_res.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    docs_stmt = select(Document).where(Document.batch_id == batch_id)
    docs_res = await db.execute(docs_stmt)
    documents = docs_res.scalars().all()

    results = []
    DocB = aliased(Document)

    for doc in documents:
        comps_stmt = (
            select(Comparison, DocB.filename)
            .join(DocB, Comparison.doc_b == DocB.id)
            .where(Comparison.doc_a == doc.id)
            .order_by(Comparison.similarity.desc())
        )
        comps_res = await db.execute(comps_stmt)
        comps = comps_res.all()

        plagiarism_details = []
        for comp, match_filename in comps:
            plagiarism_details.append(
                {
                    "similar_document": match_filename,
                    "similarity": comp.similarity,
                    "matches": comp.matches or [],
                }
            )

        results.append(
            {
                "document_id": str(doc.id),
                "filename": doc.filename,
                "status": doc.status,
                "ai_analysis": {
                    "score": doc.ai_score,
                    "is_ai": doc.is_ai_generated,
                    "confidence": doc.ai_confidence,
                    "provider": doc.ai_provider,
                },
                "plagiarism_analysis": plagiarism_details,
            }
        )

    return {"status": "ok", "data": results}
