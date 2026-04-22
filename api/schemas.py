from pydantic import BaseModel
from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    confidence: Optional[float] = None