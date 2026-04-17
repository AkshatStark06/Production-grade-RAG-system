from fastapi import APIRouter
from api.schemas import QueryRequest, QueryResponse
from services.rag_service import RAGService

router = APIRouter()

# Initialize once (important for performance)
rag_service = RAGService()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_service.query(request.query)
    return result