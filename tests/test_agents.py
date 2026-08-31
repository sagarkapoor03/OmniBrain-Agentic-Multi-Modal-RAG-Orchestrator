import pytest
from app.agents.graph import run_agentic_pipeline
from app.guardrails.guardrail_manager import guardrail_manager

def test_supervisor_routing_and_sql_agent():
    """Verify Supervisor Agent routes SQL prompt correctly to SQL Agent."""
    query = "What was NVIDIA's highest stock closing price in 2024?"
    res = run_agentic_pipeline(query)

    assert res["is_in_scope"] is True
    assert res["route"] == "sql_agent"
    assert res["sql_query"] is not None
    assert "SELECT" in res["sql_query"].upper()
    assert "145.89" in res["final_answer"]

def test_guardrail_refusal():
    """Verify Financial Scope Guardrail blocks out-of-scope prompts."""
    query = "How do I bake a delicious chocolate cake?"
    res = run_agentic_pipeline(query)

    assert res["is_in_scope"] is False
    assert "refusal" in res["route"]
    assert "strictly configured" in res["final_answer"]
