from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    doc_name: Optional[str]
    route: Optional[str]
    
    # Retrieved evidence
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_images: List[Dict[str, Any]]
    
    # Structured SQL
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    
    # Self-RAG & Guardrails
    self_rag_eval_count: int
    relevance_score: Optional[str]  # "relevant" or "irrelevant"
    is_grounded: Optional[bool]
    is_in_scope: Optional[bool]
    refusal_reason: Optional[str]
    
    # Response synthesis & citations
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]
    
    # Safe progress status for Streamlit UI
    execution_steps: List[str]
