from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.search_agent import search_agent_node
from app.agents.vision_agent import vision_agent_node
from app.agents.sql_agent import sql_agent_node
from app.agents.self_rag import self_rag_evaluator_node, query_rewriter_node
from app.agents.synthesis import synthesis_node
from app.guardrails.guardrail_manager import guardrail_manager

def route_supervisor(state: AgentState) -> str:
    """Routing function following Supervisor Agent decision."""
    r = state.get("route", "search_agent")
    if r in ["search_agent", "vision_agent", "sql_agent", "synthesis"]:
        return r
    return "search_agent"

def route_self_rag(state: AgentState) -> str:
    """Routing function following Self-RAG evaluation."""
    score = state.get("relevance_score", "relevant")
    eval_count = state.get("self_rag_eval_count", 0)

    if score == "irrelevant" and eval_count < 2:
        return "query_rewriter"
    return "synthesis"

def build_workflow():
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("vision_agent", vision_agent_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("self_rag_evaluator", self_rag_evaluator_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("synthesis", synthesis_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Add Conditional Edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "search_agent": "search_agent",
            "vision_agent": "vision_agent",
            "sql_agent": "sql_agent",
            "synthesis": "synthesis"
        }
    )

    # Edge from Search Agent to Self-RAG Evaluator
    workflow.add_edge("search_agent", "self_rag_evaluator")

    # Conditional Edge from Self-RAG Evaluator
    workflow.add_conditional_edges(
        "self_rag_evaluator",
        route_self_rag,
        {
            "query_rewriter": "query_rewriter",
            "synthesis": "synthesis"
        }
    )

    # Edge from Query Rewriter back to Search Agent
    workflow.add_edge("query_rewriter", "search_agent")

    # Edges to Synthesis
    workflow.add_edge("vision_agent", "synthesis")
    workflow.add_edge("sql_agent", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()

app_graph = build_workflow()

def run_agentic_pipeline(query: str, doc_name: str = "NVIDIA Financial Report") -> AgentState:
    """Executes the full OmniBrain agentic pipeline for a user query."""

    # 1. Financial Domain Scope Guardrail Check
    is_in_scope, refusal_reason = guardrail_manager.check_financial_scope(query)
    if not is_in_scope:
        return {
            "query": query,
            "rewritten_query": None,
            "doc_name": doc_name,
            "route": "refusal",
            "retrieved_chunks": [],
            "retrieved_images": [],
            "sql_query": None,
            "sql_result": None,
            "self_rag_eval_count": 0,
            "relevance_score": None,
            "is_grounded": True,
            "is_in_scope": False,
            "refusal_reason": refusal_reason,
            "final_answer": refusal_reason,
            "citations": [],
            "execution_steps": ["Financial Scope Guardrail triggered: Request rejected as out-of-scope."]
        }

    # Initial State
    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "doc_name": doc_name,
        "route": None,
        "retrieved_chunks": [],
        "retrieved_images": [],
        "sql_query": None,
        "sql_result": None,
        "self_rag_eval_count": 0,
        "relevance_score": None,
        "is_grounded": True,
        "is_in_scope": True,
        "refusal_reason": None,
        "final_answer": None,
        "citations": [],
        "execution_steps": ["Agent Orchestration initialized."]
    }

    final_state = app_graph.invoke(initial_state)
    return final_state
