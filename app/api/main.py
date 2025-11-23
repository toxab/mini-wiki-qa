"""FastAPI application with health check and RAG endpoints"""
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from core.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mini-Wiki Q&A API",
    description="RAG-based question answering system",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request/Response Models ===

class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_rerank: Optional[bool] = Field(default=False, description="Enable reranking")


class Citation(BaseModel):
    """Citation for a retrieved chunk"""
    document: str = Field(..., description="Document name")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default=[], description="Retrieved chunks")
    metadata: dict = Field(default={}, description="Request metadata")


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    timestamp: str
    services: dict


# === Security ===

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verify API key from header"""
    if x_api_key != settings.API_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# === Endpoints ===

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Mini-Wiki Q&A API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint - verify all services"""
    import httpx

    services = {}

    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.QDRANT_URL}/readyz", timeout=2.0)
            services["qdrant"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)}"

    # Check LLM backend
    try:
        llm_url = settings.LM_STUDIO_URL if settings.LLM_BACKEND == "lm-studio" else settings.OLLAMA_URL
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{llm_url}/models", timeout=2.0)
            services["llm"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["llm"] = f"error: {str(e)}"

    # Check MLflow
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health", timeout=2.0)
            services["mlflow"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["mlflow"] = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if all(v == "ok" for v in services.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        services=services
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(
        request: AskRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    Ask a question and get an answer with citations
    """
    try:
        logger.info(f"Received query: {request.query[:50]}...")

        # Import here to avoid startup delays
        from rag.retrieval import get_retriever
        from rag.generation import get_generator
        from rag.reranker import get_reranker

        # 1. Retrieve relevant chunks
        retriever = get_retriever()
        chunks = retriever.retrieve(request.query, top_k=request.top_k)

        # 2. Rerank if requested
        if request.use_rerank:
            logger.info("Applying reranking...")
            reranker = get_reranker()
            chunks = reranker.rerank(request.query, chunks, top_k=request.top_k)

        # 3. Generate answer
        generator = get_generator()
        answer = generator.generate(request.query, chunks)

        # 4. Format citations
        citations = [
            Citation(
                document=chunk["source"].split("/")[-1],  # Extract filename
                chunk_id=f"chunk_{idx}",
                text=chunk["text"][:200] + "...",  # Preview
                score=chunk.get("rerank_score", chunk.get("score", 0.0))  # Use rerank score if available
            )
            for idx, chunk in enumerate(chunks)
        ]

        return AskResponse(
            answer=answer,
            citations=citations,
            metadata={
                "query": request.query,
                "top_k": request.top_k,
                "use_rerank": request.use_rerank,
                "llm_backend": settings.LLM_BACKEND,
                "chunks_retrieved": len(chunks)
            }
        )

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/ingest", tags=["Admin"])
async def ingest_documents(
        api_key: str = Depends(verify_api_key)
):
    """
    Trigger document ingestion

    Implement document processing pipeline
    """
    try:
        # TODO: Implement ingestion
        # 1. Load documents from data/documents/
        # 2. Chunk text
        # 3. Generate embeddings
        # 4. Index in Qdrant

        return {
            "status": "success",
            "message": "Ingestion not yet implemented."
        }

    except Exception as e:
        logger.error(f"Error in /ingest: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# === Startup/Shutdown Events ===

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Mini-Wiki Q&A API...")
    logger.info(f"LLM Backend: {settings.LLM_BACKEND}")
    logger.info(f"Qdrant URL: {settings.QDRANT_URL}")
    logger.info(f"MLflow URI: {settings.MLFLOW_TRACKING_URI}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Mini-Wiki Q&A API...")