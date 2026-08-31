import os
import pytest
from app.database.stock_db import init_db, run_sql_query
from app.database.seed_stock_db import seed_all

def test_database_initialization_and_seeding():
    """Verify stock_prices and company_financials table seeding for NVDA."""
    init_db()
    seed_all()

    # Query stock_prices count
    prices = run_sql_query("SELECT COUNT(*) as count FROM stock_prices WHERE symbol='NVDA';")
    assert len(prices) > 0
    assert prices[0]["count"] > 0

    # Query company_financials
    financials = run_sql_query("SELECT symbol, fiscal_year, revenue_millions FROM company_financials WHERE symbol='NVDA' ORDER BY fiscal_year DESC;")
    assert len(financials) > 0
    assert financials[0]["revenue_millions"] == 130400.0

def test_sql_query_execution():
    """Test SQL query for highest stock closing price in 2024."""
    query = "SELECT MAX(close) as max_close FROM stock_prices WHERE symbol='NVDA' AND date LIKE '2024%';"
    res = run_sql_query(query)
    assert len(res) > 0
    assert "max_close" in res[0]
    assert res[0]["max_close"] > 0
