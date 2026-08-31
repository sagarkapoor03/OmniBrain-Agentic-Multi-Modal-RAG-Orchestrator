import os
import sqlite3
import pandas as pd
from app.config import settings
from app.database.stock_db import init_db, get_db_connection

def seed_nvda_stock_prices():
    """Reads data/stock_data/NVDA.csv and loads it idempotently into stock_prices table."""
    csv_path = settings.NVDA_CSV_PATH
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV file not found at {csv_path}. Skipping stock price seeding.")
        return 0

    print(f"Reading NVIDIA historical stock data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean & normalize column names
    # Handled column names: Date, Adj Close, Close, High, Low, Open, Volume
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns:
        raise ValueError("CSV missing required 'Date' column")

    # Format date column to ISO YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["symbol"] = "NVDA"

    # Map column names if needed
    col_map = {
        "adj_close": "adj_close",
        "close": "close",
        "high": "high",
        "low": "low",
        "open": "open",
        "volume": "volume"
    }

    conn = get_db_connection()
    cursor = conn.cursor()

    inserted_count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["symbol"],
            row["date"],
            float(row.get("open", 0.0)),
            float(row.get("high", 0.0)),
            float(row.get("low", 0.0)),
            float(row.get("close", 0.0)),
            float(row.get("adj_close", row.get("close", 0.0))),
            int(row.get("volume", 0))
        ))
        inserted_count += 1

    conn.commit()
    conn.close()
    print(f"Successfully seeded {inserted_count} NVDA stock price rows into database.")
    return inserted_count


def seed_nvda_financials():
    """Seeds NVIDIA official annual financial figures (FY2021 - FY2025)."""
    financials = [
        # NVIDIA Official Fiscal Year Data (FY ends late January)
        ("NVDA", 2021, "FY", 16675.0, 4332.0, 4532.0, 1.73),
        ("NVDA", 2022, "FY", 26914.0, 9752.0, 10041.0, 3.85),
        ("NVDA", 2023, "FY", 26974.0, 4368.0, 4224.0, 1.74),
        ("NVDA", 2024, "FY", 60922.0, 29760.0, 32972.0, 11.93),
        ("NVDA", 2025, "FY", 130400.0, 72800.0, 81500.0, 2.95),  # Post 10-for-1 stock split EPS
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    for item in financials:
        cursor.execute("""
            INSERT OR REPLACE INTO company_financials 
            (symbol, fiscal_year, quarter, revenue_millions, net_income_millions, operating_income_millions, eps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, item)

    conn.commit()
    conn.close()
    print(f"Successfully seeded {len(financials)} NVIDIA annual financial rows into database.")


def seed_all():
    init_db()
    seed_nvda_stock_prices()
    seed_nvda_financials()

if __name__ == "__main__":
    seed_all()
