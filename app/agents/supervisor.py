from app.agents.state import AgentState
from app.agents.llm import llm_client

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor Agent: Analyzes user query and decides routing path."""
    query = state["query"]
    
    prompt = f"""You are the Supervisor Orchestrator for OmniBrain, an AI system analyzing NVIDIA (NVDA) multi-modal documents and stock databases.
Analyze the user query and select the SINGLE best specialized agent to execute next:

Options:
- 'search_agent': Use when query asks about textual document contents, financial narrative, segment performance, risks, or general PDF report text.
- 'vision_agent': Use when query asks about charts, figures, visual trends, diagrams, or visual revenue graphs.
- 'sql_agent': Use when query asks about numerical stock price history, closing prices, historical volume, daily highs/lows, or tabular database statistics.
- 'synthesis': Use when sufficient evidence has already been retrieved.

User Query: "{query}"

Select next agent (respond ONLY with one of: search_agent, vision_agent, sql_agent, synthesis):"""

    route_decision = llm_client.generate_text(prompt).strip().lower()

    # Validate decision
    valid_routes = ["search_agent", "vision_agent", "sql_agent", "synthesis"]
    if route_decision not in valid_routes:
        # Fallback routing heuristic
        if any(k in query.lower() for k in ["stock", "price", "closing", "high", "volume", "2024", "1999", "2023"]):
            route_decision = "sql_agent"
        elif any(k in query.lower() for k in ["chart", "figure", "visual", "graph"]):
            route_decision = "vision_agent"
        else:
            route_decision = "search_agent"

    state["route"] = route_decision
    steps = state.get("execution_steps", [])
    steps.append(f"Supervisor routed query to: `{route_decision}`")
    state["execution_steps"] = steps
    
    return state
