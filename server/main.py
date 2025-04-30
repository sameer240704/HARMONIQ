# main.py
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
import os
from data_fetcher import fetch_fintech_news
from data_types import BankDataInput, AnalysisRequest
from analysis import analyze_bank_performance, predict_future_trends, calculate_digital_adoption_metrics, compare_banks_technology_spending

app = FastAPI(title="Banking IT Impact Analysis",
              description="API for analyzing the role of IT in banking sector",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE = os.path.join(DATABASE_DIR, "banking_it.db")

def init_db():
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

init_db()

def map_bank_to_ticker(bank_name):
    """
    Map bank names to their Yahoo Finance ticker symbols
    """
    bank_tickers = {
        "IDFC First Bank": "IDFCFIRSTB.NS",
        "Wells Fargo & Company": "WFC",
        "Punjab National Bank": "PNB.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "Canara Bank": "CANBK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "SBI": "SBIN.NS"
    }
    return bank_tickers.get(bank_name)

def generate_synthetic_it_spending(hist_data):
    """
    Generate synthetic IT spending data based on stock performance
    """
    base = np.random.uniform(100, 500)
    trend = hist_data['Close'].pct_change().cumsum().fillna(0)
    noise = np.random.normal(0, 0.1, len(hist_data))
    return (base * (1 + trend * 0.5 + noise)).values

def generate_synthetic_digital_transactions(hist_data):
    """
    Generate synthetic digital transactions data
    """
    base = np.random.uniform(10000, 50000)
    trend = np.linspace(0, 1, len(hist_data)) * 2
    seasonality = np.sin(np.linspace(0, 10*np.pi, len(hist_data))) * 0.2
    noise = np.random.normal(0, 0.1, len(hist_data))
    return (base * (1 + trend + seasonality + noise)).astype(int)

def generate_synthetic_mobile_users(hist_data):
    """
    Generate synthetic mobile app users data
    """
    base = np.random.uniform(5000, 20000)
    trend = np.linspace(0, 1.5, len(hist_data))
    noise = np.random.normal(0, 0.05, len(hist_data))
    return (base * (1 + trend + noise)).astype(int)

def store_bank_data(bank_name, hist_data):
    """
    Store bank data in SQLite database
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM bank_data WHERE bank_name = ?", (bank_name,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Data for {bank_name} already exists in database. Skipping insertion.")
        conn.close()
        return
    
    for date, row in hist_data.iterrows():
        cursor.execute('''
            INSERT INTO bank_data 
            (bank_name, date, stock_price, digital_transactions, it_spending, mobile_users)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            bank_name,
            date.strftime('%Y-%m-%d'),
            row['Close'],
            row['Digital_Transactions'],
            row['IT_Spending'],
            row['Mobile_Users']
        ))
    
    conn.commit()
    conn.close()
    print(f"Stored data for {bank_name} in database.")

def fetch_and_store_bank_data(bank_name, period="10y"):
    """
    Fetch bank data from Yahoo Finance and store in database
    """
    try:
        ticker = map_bank_to_ticker(bank_name)
        if not ticker:
            print(f"No ticker found for {bank_name}")
            return None
    
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            print(f"No historical data found for {bank_name} ({ticker})")
            return None
        
        hist['IT_Spending'] = generate_synthetic_it_spending(hist)
        hist['Digital_Transactions'] = generate_synthetic_digital_transactions(hist)
        hist['Mobile_Users'] = generate_synthetic_mobile_users(hist)
        
        store_bank_data(bank_name, hist)
        
        return hist
    except Exception as e:
        print(f"Error fetching data for {bank_name}: {str(e)}")
        return None

def ensure_data_in_db(bank_names):
    """
    Check if data exists in database for given banks, if not, fetch and store it
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    for bank in bank_names:
        cursor.execute("SELECT COUNT(*) FROM bank_data WHERE bank_name = ?", (bank,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            print(f"No data found for {bank}. Fetching from Yahoo Finance...")
            fetch_and_store_bank_data(bank)
    
    conn.close()

@app.on_event("startup")
def load_initial_data():
    """
    Load data for default banks on startup
    """
    default_banks = ["IDFC First Bank", "SBI", "Punjab National Bank", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank", "Kotak Mahindra Bank"]
    ensure_data_in_db(default_banks)

@app.get("/")
def read_root():
    return {"message": "Banking IT Impact Analysis API"}

@app.post("/add-bank-data")
async def add_bank_data(data: BankDataInput):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO bank_data (bank_name, date, stock_price, digital_transactions, it_spending, mobile_users)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data.bank_name, data.date, data.stock_price, data.digital_transactions, data.it_spending, data.mobile_users))
    conn.commit()
    conn.close()
    return {"message": "Data added successfully"}

@app.get("/fetch-market-data")
async def fetch_market_data(bank_names: List[str] = Query(...), background_tasks: BackgroundTasks = None):
    try:
        # First ensure data exists in database
        ensure_data_in_db(bank_names)
        
        # Fetch data from database
        conn = sqlite3.connect(DATABASE)
        data = {}
        
        for bank in bank_names:
            query = "SELECT * FROM bank_data WHERE bank_name = ? ORDER BY date"
            df = pd.read_sql(query, conn, params=[bank])
            
            if not df.empty:
                data[bank] = {
                    'historical': df.to_dict('records')
                }
        
        conn.close()
        
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/fetch-fintech-news")
async def get_fintech_news(keywords: List[str] = Query(["digital banking", "fintech", "banking technology"])):
    try:
        news = fetch_fintech_news(keywords)
        return {"news": news}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-bank-performance")
async def analyze_performance(request: AnalysisRequest):
    try:
        # First ensure data exists in database
        ensure_data_in_db(request.bank_names)
        
        conn = sqlite3.connect(DATABASE)
        query = f'''
            SELECT * FROM bank_data 
            WHERE bank_name IN ({','.join(['?']*len(request.bank_names))})
            AND date BETWEEN ? AND ?
        '''
        params = request.bank_names + [request.start_date, request.end_date]
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given parameters")
        
        results = analyze_bank_performance(df, request.metrics)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict-future-trends")
async def predict_trends(bank_name: str, metric: str, periods: int = 365):
    try:
        ensure_data_in_db([bank_name])
        
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT date, {} as value FROM bank_data 
            WHERE bank_name = ?
            ORDER BY date
        '''.format(metric)
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given bank and metric")
        
        forecast = predict_future_trends(df, periods)
        return forecast
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/compare-technology-spending")
async def compare_technology_spending(bank_names: List[str] = Query(...)):
    try:
        # Ensure data exists in database
        ensure_data_in_db(bank_names)
        
        # Fetch data from database
        conn = sqlite3.connect(DATABASE)
        query = f'''
            SELECT bank_name, AVG(it_spending) as avg_it_spending, 
                   AVG(digital_transactions) as avg_digital_transactions,
                   AVG(mobile_users) as avg_mobile_users
            FROM bank_data 
            WHERE bank_name IN ({','.join(['?']*len(bank_names))})
            GROUP BY bank_name
        '''
        df = pd.read_sql(query, conn, params=bank_names)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given banks")
        
        comparison = compare_banks_technology_spending(df)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/digital-adoption-metrics")
async def digital_adoption_metrics(bank_name: str):
    try:
        # Ensure data exists in database
        ensure_data_in_db([bank_name])
        
        # Fetch data from database
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT date, digital_transactions, mobile_users, it_spending
            FROM bank_data 
            WHERE bank_name = ?
            ORDER BY date
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given bank")
        
        metrics = calculate_digital_adoption_metrics(df)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)