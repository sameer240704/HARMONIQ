from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import plotly.graph_objects as go
import sqlite3, os, httpx, nltk, json, logging
from datetime import datetime, timedelta
from data_fetcher import fetch_fintech_news
from data_types import BankDataInput, AnalysisRequest
from analysis import analyze_bank_performance, create_stock_trend_chart, create_correlation_chart, handle_json
from nltk.sentiment import SentimentIntensityAnalyzer
from tavily import TavilyClient
import numpy as np
import yfinance as yf
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

nltk.download('vader_lexicon')

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('data_loading.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sia = SentimentIntensityAnalyzer()

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE = os.path.join(DATABASE_DIR, "banking_it.db")


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS bank_analysis")
    
    cursor.execute('''
        CREATE TABLE bank_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT,
            date TEXT,
            stock_price REAL,
            volume REAL,
            pe_ratio REAL,
            rnd_expenses REAL,
            it_news_count INTEGER,
            it_sentiment REAL,
            digital_mentions INTEGER,
            cloud_adoption_score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def map_bank_to_ticker(bank_name: str) -> Optional[str]:
    """Map bank names to their ticker symbols"""
    bank_tickers = {
        "IDFC First Bank": "IDFCFIRSTB.BO",
        "SBI": "SBIN.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS"
    }
    return bank_tickers.get(bank_name)

async def fetch_financial_metrics(ticker: str) -> dict:
    """Fetch financial metrics from Alpha Vantage"""
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
    return {
        'pe_ratio': float(data.get('PERatio', 0)),
        'rnd_expenses': float(data.get('ResearchAndDevelopment', 0))
    }

async def fetch_news_analysis(bank_name: str, date: datetime, max_articles: int = 20) -> dict:
    """Fetch and analyze IT-related news using Tavily"""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    query = f"{bank_name} IT initiatives OR digital transformation"
    
    start_date = (date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (date + timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_articles,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            start_date=start_date,
            end_date=end_date
        )
        articles = response.get('results', [])
        
        relevant_articles = []
        for article in articles:
            try:
                article_date = datetime.strptime(article.get('published_date', ''), "%Y-%m-%d").date()
                if article_date == date.date():
                    relevant_articles.append(article)
            except (ValueError, AttributeError):
                continue
        
        if not relevant_articles:
            return {
                'it_news_count': 0,
                'it_sentiment': 0.0,
                'digital_mentions': 0
            }
        
        sentiments = []
        digital_mentions = 0
        for article in relevant_articles:
            content = article.get('content', '')
            if not content:
                continue
                
            sentiment = sia.polarity_scores(content)['compound']
            sentiments.append(sentiment)
            
            digital_mentions += ('digital' in content.lower())
        
        return {
            'it_news_count': len(relevant_articles),
            'it_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'digital_mentions': digital_mentions
        }
    except Exception as e:
        print(f"News fetch error: {str(e)}")
        return {
            'it_news_count': 0,
            'it_sentiment': 0.0,
            'digital_mentions': 0
        }
    
async def fetch_cloud_adoption(ticker: str) -> float:
    """Fetch cloud adoption metrics using Alpha Vantage earnings calls"""
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        content = response.text
    
    cloud_terms = ['cloud', 'aws', 'azure', 'google cloud', 'digital infrastructure']
    return sum(content.lower().count(term) for term in cloud_terms) / 10

async def fetch_and_store_bank_data(bank_name: str):
    """Main data fetching and processing function"""
    try:
        logger.info(f"Starting data fetch for {bank_name}")

        max_days = 365  
        max_news_articles = 20

        ticker = map_bank_to_ticker(bank_name)
        if not ticker:
            logger.warning(f"No ticker found for {bank_name}")
            return False

        logger.info(f"Fetching stock data for {bank_name} ({ticker})")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{max_days}d")
        
        if hist.empty:
            logger.warning(f"No historical data found for {bank_name}")
            return False

        logger.info(f"Fetching financial metrics for {bank_name}")
        financials = await fetch_financial_metrics(ticker)
        logger.info(f"Fetching cloud adoption metrics for {bank_name}")
        cloud_score = await fetch_cloud_adoption(ticker)
        
        df = hist[['Close', 'Volume']].copy()
        df = df.rename(columns={
            'Close': 'stock_price',
            'Volume': 'volume'
        }).reset_index()
        
        df = df.rename(columns={'Date': 'date'})
        
        monthly_df = df.iloc[::30].copy()
        
        news_data = []
        for _, row in monthly_df.iterrows():
            news = await fetch_news_analysis(
                bank_name, 
                row['date'],  
                max_articles=max_news_articles  
            )
            news_data.append(news)
        
        news_df = pd.DataFrame(news_data)
        
        df = pd.merge(
            df,
            monthly_df[['date']].join(news_df),
            on='date',
            how='left'
        ).fillna(0)

        df['pe_ratio'] = financials['pe_ratio']
        df['rnd_expenses'] = financials['rnd_expenses']
        df['cloud_adoption_score'] = cloud_score
        df['bank_name'] = bank_name
        
        required_columns = [
            'bank_name', 'date', 'stock_price', 'volume',
            'pe_ratio', 'rnd_expenses', 'it_news_count',
            'it_sentiment', 'digital_mentions', 'cloud_adoption_score'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # or appropriate default value
                
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0 if col == 'it_sentiment' else 0 
            df[col] = df[col].fillna(0).astype(float if 'sentiment' in col else int)
        
        logger.info(f"Storing data for {bank_name} in database")
        conn = sqlite3.connect(DATABASE)
        df.to_sql('bank_analysis', conn, if_exists='append', index=False)
        conn.close()

        record_count = len(df)
        logger.info(f"Successfully stored {record_count} records for {bank_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {bank_name}: {str(e)}", exc_info=True)
        return False
        
@app.on_event("startup")
async def initialize_data():
    """Initialize data for default banks"""
    default_banks = ["IDFC First Bank", "SBI", "HDFC Bank", "ICICI Bank", "Axis Bank"]
    logger.info(f"Starting data initialization for {len(default_banks)} banks")

    for bank in default_banks:
        logger.info(f"Processing bank: {bank}")
        await fetch_and_store_bank_data(bank)

# default_banks = ["IDFC First Bank", "SBI", "Punjab National Bank", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank", "Kotak Mahindra Bank"]

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

@app.get("/financial-metrics")
async def get_financial_metrics(bank_name: str):
    """Get stored financial metrics for analysis"""
    conn = sqlite3.connect(DATABASE)
    query = '''
        SELECT 
            date, 
            stock_price, 
            pe_ratio, 
            rnd_expenses, 
            COALESCE(it_news_count, 0) as it_news_count,
            COALESCE(it_sentiment, 0) as it_sentiment,
            digital_mentions, 
            cloud_adoption_score
        FROM bank_analysis
        WHERE bank_name = ?
        ORDER BY date DESC
        LIMIT 365
    '''
    df = pd.read_sql(query, conn, params=[bank_name])
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not available")
    
    return {
        "timeline": df.to_dict(orient='records'),
        "stats": {
            "avg_it_news": df.get('it_news_count', pd.Series([0])).mean(),
            "avg_sentiment": df.get('it_sentiment', pd.Series([0])).mean(),
            "rnd_growth": df.get('rnd_expenses', pd.Series([0])).pct_change().mean(),
            "cloud_adoption": df.get('cloud_adoption_score', pd.Series([0])).iloc[0]
        }
    }

@app.get("/compare-banks")
async def compare_banks(bank_names: List[str] = Query(...)):
    """Compare multiple banks' IT metrics"""
    conn = sqlite3.connect(DATABASE)
    placeholders = ','.join(['?']*len(bank_names))
    query = f'''
        SELECT bank_name, 
               AVG(rnd_expenses) as avg_rnd,
               SUM(it_news_count) as total_it_news,
               AVG(cloud_adoption_score) as cloud_score
        FROM bank_analysis
        WHERE bank_name IN ({placeholders})
        GROUP BY bank_name
    '''
    df = pd.read_sql(query, conn, params=bank_names)
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    return df.to_dict(orient='records')

@app.get("/it-impact-analysis")
async def it_impact_analysis(bank_name: str):
    """Comprehensive IT impact analysis report"""
    conn = sqlite3.connect(DATABASE)
    query = '''
        SELECT * FROM bank_analysis
        WHERE bank_name = ?
        ORDER BY date DESC
    '''
    df = pd.read_sql(query, conn, params=[bank_name])
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Bank data not found")
    
    analysis = {
        "stock_correlation": df[['stock_price', 'it_news_count']].corr().iloc[0,1],
        "rnd_growth": df['rnd_expenses'].pct_change().mean(),
        "digital_trend": df['digital_mentions'].pct_change().mean(),
        "cloud_adoption": df['cloud_adoption_score'].iloc[0],
        "news_sentiment_trend": df['it_sentiment'].rolling(30).mean().iloc[-1]
    }
    
    return {
        "metadata": {
            "bank": bank_name,
            "analysis_period": f"{df['date'].min()} to {df['date'].max()}"
        },
        "metrics": analysis,
        "raw_data": df.to_dict(orient='records')
    }

@app.get("/fetch-fintech-news")
async def get_fintech_news(keywords: List[str] = Query(["digital banking", "fintech", "banking technology"])):
    try:
        news = fetch_fintech_news(keywords)
        return {"news": news}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)