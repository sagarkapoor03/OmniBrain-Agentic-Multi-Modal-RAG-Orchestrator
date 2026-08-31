from app.agents.state import AgentState
from app.retrieval.vector_store import vector_store

def search_agent_node(state: AgentState) -> AgentState:
    """Search Agent: Performs vector retrieval against Qdrant for document text chunks."""
    query = state.get("rewritten_query") or state["query"]
    
    retrieved = vector_store.search(query=query, top_k=4)
    state["retrieved_chunks"] = retrieved

    # Build citations list
    citations = state.get("citations", [])
    for item in retrieved:
        citations.append({
            "type": "document_chunk",
            "doc_name": item.get("doc_name", "Report"),
            "page_number": item.get("page_number", 1),
            "snippet": item.get("text", "")[:150] + "..."
        })
    state["citations"] = citations

    steps = state.get("execution_steps", [])
    steps.append(f"Search Agent retrieved {len(retrieved)} document chunks from Qdrant vector store.")
    state["execution_steps"] = steps

    return state
