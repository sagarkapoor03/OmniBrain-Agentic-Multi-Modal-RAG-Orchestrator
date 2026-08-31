from app.agents.state import AgentState
from app.agents.llm import llm_client
from app.database.stock_db import run_sql_query

def sql_agent_node(state: AgentState) -> AgentState:
    """SQL Agent: Converts natural language query to SQLite query and executes it against stocks database."""
    query = state["query"]

    schema_info = """
Database Schema:
1. stock_prices (symbol VARCHAR, date DATE, open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER)
   - Contains daily stock market price history for symbol 'NVDA' from 1999 to 2025.
   - Date format is ISO 'YYYY-MM-DD'.

2. company_financials (symbol VARCHAR, fiscal_year INT, quarter VARCHAR, revenue_millions REAL, net_income_millions REAL, operating_income_millions REAL, eps REAL)
   - Contains NVIDIA corporate annual financial metrics for symbol 'NVDA' (FY2021 to FY2025).
"""

    prompt = f"""You are an expert Text-to-SQL Agent for OmniBrain.
Given the database schema below and user query, generate a SINGLE valid read-only SQLite SQL query.
Return ONLY the raw SQL query string without markdown code fences or explanation.

{schema_info}

User Query: "{query}"

SQL Query:"""

    raw_sql = llm_client.generate_text(prompt).strip()
    
    # Clean code fences if present
    if raw_sql.startswith("```"):
        raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    # Safety check: enforce read-only SELECT queries
    if not raw_sql.upper().startswith("SELECT"):
        raw_sql = "SELECT symbol, date, close, volume FROM stock_prices WHERE symbol='NVDA' ORDER BY date DESC LIMIT 5;"

    sql_results = run_sql_query(raw_sql)

    state["sql_query"] = raw_sql
    state["sql_result"] = sql_results

    citations = state.get("citations", [])
    citations.append({
        "type": "sql_database",
        "query": raw_sql,
        "row_count": len(sql_results),
        "snippet": f"SQL Query: `{raw_sql}` ({len(sql_results)} rows returned)"
    })
    state["citations"] = citations

    steps = state.get("execution_steps", [])
    steps.append(f"SQL Agent generated and executed query: `{raw_sql}` ({len(sql_results)} rows).")
    state["execution_steps"] = steps

    return state
