import sqlite3
from typing import List, Dict, Any
from app.config import settings

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.STOCK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create stock_prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
    """)

    # Create company_financials table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_financials (
            symbol TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            quarter TEXT NOT NULL,
            revenue_millions REAL,
            net_income_millions REAL,
            operating_income_millions REAL,
            eps REAL,
            PRIMARY KEY (symbol, fiscal_year, quarter)
        );
    """)

    conn.commit()
    conn.close()

def run_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute a read-only SQL query against the stocks database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        return result
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()
