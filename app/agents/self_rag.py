from app.agents.state import AgentState
from app.agents.llm import llm_client

def self_rag_evaluator_node(state: AgentState) -> AgentState:
    """Self-RAG Evaluator: Evaluates if retrieved evidence is sufficient and relevant."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    sql_res = state.get("sql_result", [])
    images = state.get("retrieved_images", [])

    # If SQL query returned results or images were processed, consider relevant
    if sql_res and len(sql_res) > 0 and "error" not in sql_res[0]:
        state["relevance_score"] = "relevant"
        steps = state.get("execution_steps", [])
        steps.append("Self-RAG Evaluator: Retrieved SQL evidence validated as relevant.")
        state["execution_steps"] = steps
        return state

    if images and len(images) > 0:
        state["relevance_score"] = "relevant"
        steps = state.get("execution_steps", [])
        steps.append("Self-RAG Evaluator: Retrieved vision chart evidence validated as relevant.")
        state["execution_steps"] = steps
        return state

    if not chunks:
        state["relevance_score"] = "irrelevant"
        steps = state.get("execution_steps", [])
        steps.append("Self-RAG Evaluator: No chunks retrieved. Flagged for query rewriting.")
        state["execution_steps"] = steps
        return state

    context_text = "\n".join([c.get("text", "") for c in chunks[:3]])
    prompt = f"""You are the Self-RAG Context Evaluator.
Determine if the retrieved context contains useful information to answer the user query.

User Query: "{query}"

Retrieved Context:
"{context_text}"

Respond ONLY with 'relevant' if the context is adequate, or 'irrelevant' if poor/insufficient:"""

    eval_res = llm_client.generate_text(prompt).strip().lower()
    score = "relevant" if "relevant" in eval_res else "irrelevant"
    state["relevance_score"] = score

    steps = state.get("execution_steps", [])
    steps.append(f"Self-RAG Evaluator evaluated retrieval relevance: `{score}`.")
    state["execution_steps"] = steps

    return state


def query_rewriter_node(state: AgentState) -> AgentState:
    """Self-RAG Query Rewriter: Refines the search query if initial retrieval was inadequate."""
    current_count = state.get("self_rag_eval_count", 0)
    state["self_rag_eval_count"] = current_count + 1

    orig_query = state["query"]
    prompt = f"""You are a Query Rewriter for financial document RAG search.
The initial vector search query failed to retrieve sufficient relevant information.
Formulate a clearer, expanded vector search query targeting NVIDIA (NVDA) financial metrics, annual reports, revenue, or stock performance.

Original Query: "{orig_query}"

Rewritten Search Query (respond ONLY with the refined query string):"""

    new_query = llm_client.generate_text(prompt).strip()
    state["rewritten_query"] = new_query

    steps = state.get("execution_steps", [])
    steps.append(f"Self-RAG Query Rewriter generated refined query (Attempt #{current_count + 1}): `{new_query}`")
    state["execution_steps"] = steps

    return state
