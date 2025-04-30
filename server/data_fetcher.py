import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import sqlite3

def fetch_banking_data(bank_names):
    """
    Fetch financial data for banks using Yahoo Finance API
    """
    bank_data = {}
    
    for bank in bank_names:
        try:
            # Get ticker data (you may need to map bank names to their ticker symbols)
            ticker = map_bank_to_ticker(bank)
            if not ticker:
                continue
                
            stock = yf.Ticker(ticker)
            
            # Get historical market data
            hist = stock.history(period="5y")
            
            # Get financials
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Get additional info
            info = stock.info
            
            # Generate some synthetic IT-related metrics (since real data may not be available)
            it_spending = generate_synthetic_it_spending(hist)
            digital_transactions = generate_synthetic_digital_transactions(hist)
            mobile_users = generate_synthetic_mobile_users(hist)
            
            # Combine data
            hist['IT_Spending'] = it_spending
            hist['Digital_Transactions'] = digital_transactions
            hist['Mobile_Users'] = mobile_users
            hist['Bank_Name'] = bank
            
            # Store in dictionary
            bank_data[bank] = {
                'historical': hist.reset_index().to_dict('records'),
                'financials': financials.reset_index().to_dict('records'),
                'balance_sheet': balance_sheet.reset_index().to_dict('records'),
                'cashflow': cashflow.reset_index().to_dict('records'),
                'info': info
            }
            
            # Store in database
            store_bank_data(bank, hist)
            
        except Exception as e:
            print(f"Error fetching data for {bank}: {str(e)}")
            continue
            
    return bank_data

def map_bank_to_ticker(bank_name):
    """
    Map bank names to their Yahoo Finance ticker symbols
    """
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
    conn = sqlite3.connect("database/banking_it.db")
    cursor = conn.cursor()
    
    for date, row in hist_data.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO bank_data 
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

def fetch_fintech_news(keywords, max_results=10):
    """
    Fetch fintech news from NewsAPI (you'll need an API key)
    """
    API_KEY = "4f2040f14c13e0aa0110c2341861ece7"  
    base_url = "https://gnews.io/api/v4/search?"
    
    query = " OR ".join(keywords)
    date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    params = {
        'q': query,
        'from': date_from,
        'sortby': 'publishedAt',
        'lang': 'en',
        'apikey': API_KEY,
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        news_data = response.json()
    
        articles = []
        for article in news_data.get('articles', [])[:max_results]:
            articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'url': article.get('url'),
                'publishedAt': article.get('publishedAt'),
                'source': article.get('source', {}).get('name')
            })
    
        return articles
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []