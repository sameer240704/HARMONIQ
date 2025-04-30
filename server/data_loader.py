import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os

DATABASE = "database/banking_it.db"

def ensure_db_directory():
    """Ensure the database directory exists"""
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)

def init_db():
    """Initialize the database with required tables"""
    ensure_db_directory()
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bank_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT,
            date TEXT,
            stock_price REAL,
            digital_transactions INTEGER,
            it_spending REAL,
            mobile_users INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def map_bank_to_ticker(bank_name):
    """Map bank names to their Yahoo Finance ticker symbols"""
    bank_tickers = {
        "JPMorgan Chase": "JPM",
        "Bank of America": "BAC",
        "Wells Fargo": "WFC",
        "Citigroup": "C",
        "Goldman Sachs": "GS",
        "Morgan Stanley": "MS",
        "HSBC": "HSBC",
        "Barclays": "BCS",
        "Deutsche Bank": "DB",
        "HDFC Bank": "HDB",
        "ICICI Bank": "IBN",
        "SBI": "SBIN.NS"
    }
    return bank_tickers.get(bank_name)

def generate_synthetic_it_spending(hist_data):
    """Generate synthetic IT spending data based on stock performance"""
    base = np.random.uniform(100, 500)
    trend = hist_data['Close'].pct_change().cumsum().fillna(0)
    noise = np.random.normal(0, 0.1, len(hist_data))
    return (base * (1 + trend * 0.5 + noise)).values

def generate_synthetic_digital_transactions(hist_data):
    """Generate synthetic digital transactions data"""
    base = np.random.uniform(10000, 50000)
    trend = np.linspace(0, 1, len(hist_data)) * 2
    seasonality = np.sin(np.linspace(0, 10*np.pi, len(hist_data))) * 0.2
    noise = np.random.normal(0, 0.1, len(hist_data))
    return (base * (1 + trend + seasonality + noise)).astype(int)

def generate_synthetic_mobile_users(hist_data):
    """Generate synthetic mobile app users data"""
    base = np.random.uniform(5000, 20000)
    trend = np.linspace(0, 1.5, len(hist_data))
    noise = np.random.normal(0, 0.05, len(hist_data))
    return (base * (1 + trend + noise)).astype(int)

def load_sample_data():
    """Load sample data into the database for demonstration purposes"""
    # List of banks to load
    banks = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Citigroup", "Goldman Sachs"]
    
    # Initialize database
    init_db()
    
    # Check if database already has data
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bank_data")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count > 0:
        print(f"Database already has {count} records. Skipping data loading.")
        return
    
    print("Loading sample data into the database...")
    
    for bank in banks:
        print(f"Processing {bank}...")
        ticker = map_bank_to_ticker(bank)
        if not ticker:
            print(f"No ticker found for {bank}, skipping.")
            continue
        
        try:
            # Get stock data for past 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years of data
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"No historical data found for {bank} ({ticker}), skipping.")
                continue
            
            # Generate synthetic data
            it_spending = generate_synthetic_it_spending(hist)
            digital_transactions = generate_synthetic_digital_transactions(hist)
            mobile_users = generate_synthetic_mobile_users(hist)
            
            # Prepare data for database
            data_to_insert = []
            for i, (date, row) in enumerate(hist.iterrows()):
                data_to_insert.append((
                    bank,
                    date.strftime('%Y-%m-%d'),
                    row['Close'],
                    int(digital_transactions[i]),
                    float(it_spending[i]),
                    int(mobile_users[i])
                ))
            
            # Store in database
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO bank_data 
                (bank_name, date, stock_price, digital_transactions, it_spending, mobile_users)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            conn.commit()
            conn.close()
            
            print(f"Added {len(data_to_insert)} records for {bank}")
            
        except Exception as e:
            print(f"Error loading data for {bank}: {str(e)}")
    
    # Verify data was loaded
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bank_data")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Successfully loaded {count} records into the database.")

if __name__ == "__main__":
    load_sample_data()