from fastapi import APIRouter
from api.schemas import QueryRequest, QueryResponse
from services.rag_service import RAGService
from fastapi.responses import StreamingResponse
import json


router = APIRouter()

# Initialize once (important for performance)
rag_service = RAGService()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_service.query(request.query)
    return result

@router.post("/query-stream")
def query_stream(request: QueryRequest):

    def generator():
        for chunk in rag_service.stream_query(request.query):
            yield f"data: {chunk}\n\n"   # 👈 VERY IMPORTANT

    return StreamingResponse(generator(), media_type="text/event-stream")