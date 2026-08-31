from app.agents.state import AgentState
from app.agents.llm import llm_client
from app.guardrails.guardrail_manager import guardrail_manager

def synthesis_node(state: AgentState) -> AgentState:
    """Synthesis Node: Synthesizes evidence from Search, Vision, or SQL agents into a grounded response."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    images = state.get("retrieved_images", [])
    sql_query = state.get("sql_query")
    sql_result = state.get("sql_result")

    evidence_parts = []

    if chunks:
        evidence_parts.append("### Retrieved Document Text Context:")
        for idx, c in enumerate(chunks, 1):
            evidence_parts.append(f"[{idx}] Source: {c.get('doc_name')} (Page {c.get('page_number')})\nContent: {c.get('text')}")

    if images:
        evidence_parts.append("### Visual Chart Analysis Evidence:")
        for img in images:
            evidence_parts.append(f"Chart: {img.get('image_name')}\nAnalysis: {img.get('analysis')}")

    if sql_query and sql_result is not None:
        evidence_parts.append(f"### Structured SQL Query & Result:\nExecuted Query: `{sql_query}`\nRows Returned: {sql_result}")

    evidence_str = "\n\n".join(evidence_parts)

    if not evidence_str.strip():
        evidence_str = "No specific document or database evidence was retrieved."

    prompt = f"""You are OmniBrain, an expert financial analyst assistant.
Synthesize a clear, quantitative, professional response to the user query based ONLY on the supporting evidence provided below.
Provide references to specific page numbers, tables, or stock data rows where appropriate.

User Query: "{query}"

Supporting Evidence:
{evidence_str}

Synthesized Grounded Answer:"""

    answer = llm_client.generate_text(prompt)

    # Validate groundedness
    is_grounded = guardrail_manager.validate_groundedness(answer, evidence_str)
    state["is_grounded"] = is_grounded
    state["final_answer"] = answer

    steps = state.get("execution_steps", [])
    steps.append("Synthesis Node: Generated grounded multi-modal response.")
    state["execution_steps"] = steps

    return state
