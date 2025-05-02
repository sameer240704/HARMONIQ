from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Any, Dict
import pandas as pd
import sqlite3, os, nltk, json, logging
from datetime import datetime
from data_fetcher import fetch_fintech_news
from data_types import BankDataInput, AnalysisRequest
from analysis import analyze_bank_performance, create_stock_trend_chart, create_correlation_chart, handle_json
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import yfinance as yf
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import math, random
from faker import Faker
from scipy.stats import norm, skew, kurtosis
from arch import arch_model

load_dotenv()

nltk.download('punkt')           
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
        RotatingFileHandler('banking_it_analysis.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05  # 5% risk-free rate
MONTE_CARLO_SIMULATIONS = 1000
VAR_CONFIDENCE_LEVEL = 0.95

class FinancialModel:
    @staticmethod
    def gbm_model(S0, mu, sigma, days, num_paths=1):
        """Geometric Brownian Motion for stock price simulation"""
        dt = 1/252
        t = np.linspace(0, days/252, days)
        paths = np.zeros((days, num_paths))
        paths[0] = S0
        
        for i in range(1, days):
            z = np.random.normal(0, 1, num_paths)
            paths[i] = paths[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
        return paths

    @staticmethod
    def garch_model(returns, p=1, q=1):
        """GARCH model for volatility clustering"""
        model = arch_model(returns * 100, p=p, q=q)
        res = model.fit(disp='off')
        return res.conditional_volatility / 100

    @staticmethod
    def monte_carlo_var(returns, confidence_level=0.95):
        """Monte Carlo VaR calculation"""
        returns = returns.dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        mc_returns = np.random.normal(mu, sigma, MONTE_CARLO_SIMULATIONS)
        var = np.percentile(mc_returns, (1 - confidence_level) * 100)
        cvar = mc_returns[mc_returns <= var].mean()
        return var, cvar

    @staticmethod
    def calculate_risk_metrics(prices):
        """Calculate advanced risk metrics"""
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate volatility using GARCH(1,1)
        if len(returns) > 10:
            volatility = FinancialModel.garch_model(returns)
            latest_volatility = volatility.iloc[-1]
        else:
            latest_volatility = returns.std()
        
        # Calculate VaR and CVaR
        var, cvar = FinancialModel.monte_carlo_var(returns, VAR_CONFIDENCE_LEVEL)
        
        return {
            "daily_volatility": float(latest_volatility),
            "annualized_volatility": float(latest_volatility * np.sqrt(252)),
            "value_at_risk_95": float(var),
            "conditional_var_95": float(cvar),
            "returns_skewness": float(skew(returns)),
            "returns_kurtosis": float(kurtosis(returns))
        }
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        try:
            self.financial_lexicon = self._load_financial_lexicon()
        except AttributeError:
            # Fallback lexicon if financial_news isn't available
            self.financial_lexicon = set([
                'stock', 'market', 'price', 'earnings', 'revenue',
                'profit', 'loss', 'growth', 'dividend', 'investment',
                'bank', 'financial', 'economy', 'interest', 'rate'
            ])
            import warnings
            warnings.warn("Using fallback financial lexicon")

    def _load_financial_lexicon(self):
        """Load financial lexicon from NLTK or use fallback"""
        try:
            financial_words = nltk.corpus.financial_news.words()
            return set(word.lower() for word in financial_words)
        except AttributeError:
            return set()  # Return empty set if corpus not available

    def analyze(self, text):
        """Enhanced sentiment analysis with financial lexicon"""
        if not hasattr(self, 'sia'):
            self.sia = SentimentIntensityAnalyzer()
            
        base_scores = self.sia.polarity_scores(text)
        
        # Financial sentiment adjustment
        financial_terms = sum(
            1 for word in text.lower().split() 
            if word in self.financial_lexicon
        )
        adjustment = min(0.2, financial_terms * 0.02)
        
        # Contextual adjustments
        if 'growth' in text or 'profit' in text:
            adjustment += 0.1
        if 'loss' in text or 'decline' in text:
            adjustment -= 0.15
            
        # Bound the adjustment
        adjusted_score = base_scores['compound'] + adjustment
        adjusted_score = max(-1, min(1, adjusted_score))
        
        return {
            'base_score': base_scores['compound'],
            'financial_adjustment': adjustment,
            'final_score': adjusted_score,
            'positive': base_scores['pos'],
            'negative': base_scores['neg'],
            'neutral': base_scores['neu']
        }

fin_sentiment = EnhancedSentimentAnalyzer()
fake = Faker()

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE = os.path.join(DATABASE_DIR, "banking_it_impact.db")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

BANK_TICKERS = {
    "IDFC First Bank": "IDFCFIRSTB.BO",
    "SBI": "SBIN.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
}

BANK_METADATA = {
    "IDFC First Bank": {
        "base_rnd": 500000000,
        "base_it_spending": 200000000,
        "base_digital": 1000000,
        "base_mobile": 500000,
        "cloud_adoption": 0.6,
        "pe_range": (15, 25)
    },
    "SBI": {
        "base_rnd": 2000000000,
        "base_it_spending": 800000000,
        "base_digital": 5000000,
        "base_mobile": 3000000,
        "cloud_adoption": 0.5,
        "pe_range": (10, 20)
    },
    "HDFC Bank": {
        "base_rnd": 1500000000,
        "base_it_spending": 600000000,
        "base_digital": 4000000,
        "base_mobile": 2500000,
        "cloud_adoption": 0.7,
        "pe_range": (20, 30)
    },
    "ICICI Bank": {
        "base_rnd": 1200000000,
        "base_it_spending": 500000000,
        "base_digital": 3500000,
        "base_mobile": 2000000,
        "cloud_adoption": 0.65,
        "pe_range": (18, 28)
    },
    "Axis Bank": {
        "base_rnd": 800000000,
        "base_it_spending": 300000000,
        "base_digital": 2000000,
        "base_mobile": 1500000,
        "cloud_adoption": 0.55,
        "pe_range": (12, 22)
    },
    "Kotak Mahindra Bank": {
        "base_rnd": 700000000,
        "base_it_spending": 250000000,
        "base_digital": 1800000,
        "base_mobile": 1200000,
        "cloud_adoption": 0.68,
        "pe_range": (25, 35)
    }
}

IT_KEYWORDS = [
    "digital transformation", "cloud computing", "blockchain", 
    "artificial intelligence", "machine learning", "cybersecurity",
    "fintech", "API banking", "mobile banking", "core banking",
    "IT infrastructure", "data analytics", "big data", "RPA",
    "chatbots", "digital lending", "neobank", "open banking"
]

class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder to handle all special cases"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif pd.isna(obj):
            return None
        return super().default(obj)
        
    def iterencode(self, obj, _one_shot=False):
        """Special handling for regular Python floats"""
        for chunk in super().iterencode(self._fix_floats(obj), _one_shot):
            yield chunk
            
    def _fix_floats(self, obj):
        """Recursively find and fix float values"""
        if isinstance(obj, float):
            # Handle infinity and NaN values
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: self._fix_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._fix_floats(item) for item in obj]
        return obj

class EnhancedJSONResponse(JSONResponse):
    """Custom JSONResponse using EnhancedJSONEncoder"""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=EnhancedJSONEncoder,
        ).encode("utf-8")

def init_db():
    """Initialize database optimized for 5-year data"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bank_analysis'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE bank_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bank_name TEXT NOT NULL,
                date TEXT NOT NULL,
                stock_price REAL,
                volume REAL,
                pe_ratio REAL,
                rnd_expenses REAL NOT NULL CHECK(rnd_expenses > 0),
                it_spending REAL,
                digital_transactions INTEGER,
                mobile_users INTEGER,
                it_news_count INTEGER NOT NULL CHECK(it_news_count >= 0),
                it_sentiment REAL NOT NULL CHECK(it_sentiment BETWEEN -1 AND 1),
                digital_mentions INTEGER,
                cloud_adoption_score REAL,
                cybersecurity_mentions INTEGER,
                fintech_partnerships INTEGER,
                api_integrations INTEGER,
                UNIQUE(bank_name, date)
            )
        ''')
        # Optimized indexes
        cursor.execute('CREATE INDEX idx_bank_date ON bank_analysis (bank_name, date)')
        cursor.execute('CREATE INDEX idx_metrics ON bank_analysis (it_spending, digital_transactions)')
        conn.commit()
        logger.info("Database initialized for 5-year data storage")
    conn.close()

def map_bank_to_ticker(bank_name: str) -> Optional[str]:
    """Map bank names to their ticker symbols with enhanced mapping"""
    return BANK_TICKERS.get(bank_name)

def generate_synthetic_financials(bank_name: str, date: datetime, stock_price: float) -> Dict[str, float]:
    """Enhanced financial generation with stochastic processes"""
    metadata = BANK_METADATA.get(bank_name, BANK_METADATA["IDFC First Bank"])
    
    years_elapsed = (date - datetime(2020, 1, 1)).days / 365
    annual_growth = 1.15 + random.normalvariate(0, 0.02)
    growth_factor = annual_growth ** years_elapsed
    
    def ornstein_uhlenbeck(prev_value, long_term_mean, theta=0.7, sigma=0.1):
        return prev_value + theta*(long_term_mean - prev_value) + sigma*np.random.normal()
    
    it_spending = metadata["base_it_spending"] * growth_factor * (1 + 0.05 * random.normalvariate(0, 1))
    
    jump_prob = 0.05  # 5% chance of a jump
    digital_base = metadata["base_digital"] * growth_factor
    if random.random() < jump_prob:
        digital_transactions = digital_base * (1 + abs(random.normalvariate(0.2, 0.1)))
    else:
        digital_transactions = digital_base * (1 + 0.02 * random.normalvariate(0, 1))
    
    return {
        'pe_ratio': random.uniform(*metadata["pe_range"]),
        'rnd_expenses': max(0, ornstein_uhlenbeck(
            metadata["base_rnd"] * growth_factor,
            metadata["base_rnd"] * growth_factor * 1.1
        )),
        'it_spending': max(0, it_spending),
        'digital_transactions': int(abs(digital_transactions)),
        'mobile_users': int(metadata["base_mobile"] * growth_factor * (1 + 0.03 * random.normalvariate(0, 1))),
        'cloud_adoption_score': min(1.0, 
            metadata["cloud_adoption"] * (0.9 + random.random() * 0.2) * (1 + years_elapsed/5))
    }

async def fetch_financial_metrics(bank_name: str, date: datetime, stock_price: float) -> Dict[str, float]:
    """
    Generate synthetic financial metrics for the bank
    """
    return generate_synthetic_financials(bank_name, date, stock_price)

async def fetch_news_analysis(bank_name: str, date: datetime) -> Dict[str, Any]:
    """Enhanced news analysis for 5-year span"""
    year = date.year - 2020  # Years since 2020
    economic_cycle = 0.8 + 0.2 * math.sin(year * math.pi/2)  # 4-year cycle
    
    def approx_poisson(lam):
        return int(random.expovariate(1/lam))
    
    return {
        'it_news_count': int(random.triangular(5, 40, 15) * economic_cycle),
        'it_sentiment': random.normalvariate(0.2, 0.3),
        'digital_mentions': int(random.expovariate(1/15) * economic_cycle),
        'cybersecurity_mentions': approx_poisson(8 * economic_cycle),
        'fintech_partnerships': random.randint(0, 3 + year//2),
        'api_integrations': random.randint(0, 2 + year//3)
    }

async def fetch_cloud_adoption(bank_name: str) -> float:
    """
    Generate synthetic cloud adoption score
    """
    metadata = BANK_METADATA.get(bank_name, BANK_METADATA["IDFC First Bank"])
    return min(1.0, metadata["cloud_adoption"] * (0.9 + random.random() * 0.2))

async def fetch_bank_data(bank_name: str, days: int = 1825) -> pd.DataFrame:
    """Enhanced data fetcher with computational finance methods"""
    logger.info(f"Fetching data for {bank_name}")
    
    ticker = map_bank_to_ticker(bank_name)
    if not ticker:
        logger.error(f"No ticker found for {bank_name}")
        return pd.DataFrame()
    
    try:
        # Fetch stock data from yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        
        # If no historical data, use GBM simulation
        if hist.empty:
            logger.warning(f"Using GBM simulation for {bank_name}")
            initial_price = 1000  # Default starting price
            mu = 0.08  # Expected return
            sigma = 0.2  # Volatility
            simulated_prices = FinancialModel.gbm_model(initial_price, mu, sigma, days)
            dates = pd.date_range(end=datetime.now(), periods=days)
            hist = pd.DataFrame({
                'Close': simulated_prices.flatten(),
                'Volume': np.random.lognormal(mean=14, sigma=0.5, size=days)
            }, index=dates)
        
        # Prepare base DataFrame with stock data
        df = hist[['Close', 'Volume']].copy()
        df = df.rename(columns={'Close': 'stock_price', 'Volume': 'volume'})
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Calculate risk metrics
        risk_metrics = FinancialModel.calculate_risk_metrics(df['stock_price'])
        df['volatility'] = risk_metrics['daily_volatility']
        df['value_at_risk'] = risk_metrics['value_at_risk_95']
        df['bank_name'] = bank_name
        
        # Generate synthetic financial metrics with stochastic models
        financial_data = []
        for _, row in df.iterrows():
            date = datetime.strptime(row['date'], '%Y-%m-%d')
            metrics = await fetch_financial_metrics(bank_name, date, row['stock_price'])
            metrics['date'] = row['date']
            financial_data.append(metrics)
            
        financial_df = pd.DataFrame(financial_data)
        
        # Merge financial data with stock data
        df = pd.merge(df, financial_df, on='date', how='left')
        
        # Add cloud adoption score
        cloud_score = await fetch_cloud_adoption(bank_name)
        df['cloud_adoption_score'] = cloud_score
        
        # Generate synthetic news data with enhanced sentiment analysis
        news_data = []
        for date in pd.date_range(end=datetime.now(), periods=days//7, freq='7D'):
            news_content = fake.text() + " " + random.choice(IT_KEYWORDS)
            sentiment = fin_sentiment.analyze(news_content)
            
            news_entry = {
                'date': date.strftime('%Y-%m-%d'),
                'it_news_count': np.random.poisson(10),
                'it_sentiment': sentiment['final_score'],
                'digital_mentions': np.random.poisson(5),
                'cybersecurity_mentions': np.random.poisson(3),
                'fintech_partnerships': np.random.binomial(3, 0.3),
                'api_integrations': np.random.binomial(2, 0.4)
            }
            news_data.append(news_entry)
            
        news_df = pd.DataFrame(news_data)
        
        df = pd.merge(df, news_df, on='date', how='left')
        
        df = df.ffill().fillna(0)
        
        # Add additional financial metrics
        df['sharpe_ratio'] = (df['stock_price'].pct_change().mean() - RISK_FREE_RATE) / df['stock_price'].pct_change().std()
        df['sortino_ratio'] = (df['stock_price'].pct_change().mean() - RISK_FREE_RATE) / \
                             (df['stock_price'].pct_change()[df['stock_price'].pct_change() < 0].std())
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {bank_name}: {str(e)}")
        return pd.DataFrame()
    
def store_bank_data(df: pd.DataFrame):
    """Store data with additional financial metrics"""
    if df.empty:
        return False
        
    try:
        conn = sqlite3.connect(DATABASE)
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(bank_analysis)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        new_columns = ['volatility', 'value_at_risk', 'sharpe_ratio', 'sortino_ratio']
        for col in new_columns:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE bank_analysis ADD COLUMN {col} REAL")
        
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df.to_sql('bank_analysis', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error storing data: {str(e)}")
        return False

@app.on_event("startup")
async def initialize_data():
    """Initialize data for default banks only if database is empty"""
    logger.info("Checking database initialization status")
    
    init_db()
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) FROM bank_analysis 
        WHERE date >= date('now', '-30 days')
    ''')
    recent_records_count = cursor.fetchone()[0]
    conn.close()
    
    if recent_records_count > 0:
        logger.info(f"Database already contains {recent_records_count} recent records, skipping data fetch")
        return
    
    logger.info("Starting data initialization for default banks")
    
    default_banks = ["IDFC First Bank", "SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank"]

    for bank in default_banks:
        logger.info(f"Processing {bank}")
        df = await fetch_bank_data(bank, days=1825) 
        if not df.empty:
            store_bank_data(df)
            logger.info(f"Stored {len(df)} records for {bank}")
        else:
            logger.warning(f"No data fetched for {bank}")

@app.get("/")
def read_root():
    return {
        "message": "Banking IT Impact Analysis API",
        "endpoints": {
            "/financial-metrics": "Get financial metrics for a bank",
            "/compare-banks": "Compare multiple banks",
            "/it-impact-analysis": "Get comprehensive IT impact analysis",
            "/fetch-fintech-news": "Get fintech news",
            "/bank-it-trends": "Get IT adoption trends for a bank"
        }
    }

@app.get("/financial-metrics", response_class=EnhancedJSONResponse)
async def get_financial_metrics(bank_name: str):
    """Get comprehensive financial metrics with IT focus"""
    try:
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT * FROM bank_analysis
            WHERE bank_name = ?
            ORDER BY date DESC
            LIMIT 365
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Bank data not found")
        
        # Calculate key metrics - Handle potential division by zero
        try:
            digital_growth = df['digital_transactions'].pct_change().mean() * 100
            digital_growth = 0 if pd.isna(digital_growth) or np.isinf(digital_growth) else digital_growth
        except Exception:
            digital_growth = 0
            
        try:
            mobile_user_growth = df['mobile_users'].pct_change().mean() * 100
            mobile_user_growth = 0 if pd.isna(mobile_user_growth) or np.isinf(mobile_user_growth) else mobile_user_growth
        except Exception:
            mobile_user_growth = 0
            
        try:
            it_news_trend = df['it_news_count'].pct_change().mean() * 100
            it_news_trend = 0 if pd.isna(it_news_trend) or np.isinf(it_news_trend) else it_news_trend
        except Exception:
            it_news_trend = 0
        
        metrics = {
            "avg_stock_price": float(df['stock_price'].mean()) if not pd.isna(df['stock_price'].mean()) else 0.0,
            "avg_pe_ratio": float(df['pe_ratio'].mean()) if not pd.isna(df['pe_ratio'].mean()) else 0.0,
            "total_rnd": float(df['rnd_expenses'].sum()) if not pd.isna(df['rnd_expenses'].sum()) else 0.0,
            "avg_it_spending": float(df['it_spending'].mean()) if not pd.isna(df['it_spending'].mean()) else 0.0,
            "digital_growth": float(digital_growth),
            "mobile_user_growth": float(mobile_user_growth),
            "it_news_trend": float(it_news_trend),
            "cloud_adoption": float(df['cloud_adoption_score'].iloc[0]) if not pd.isna(df['cloud_adoption_score'].iloc[0]) else 0.0
        }
        
        # Clean timeline data
        timeline_df = df[[
            'date', 'stock_price', 'pe_ratio', 'rnd_expenses',
            'it_spending', 'digital_transactions', 'mobile_users',
            'it_news_count', 'cloud_adoption_score'
        ]].copy()
        
        # Replace NaN, Inf with None/null for JSON serialization
        timeline_df = timeline_df.replace([np.inf, -np.inf], np.nan)
        timeline_df = timeline_df.fillna(0)
        
        return {
            "bank": bank_name,
            "metrics": metrics,
            "timeline": timeline_df.to_dict(orient='records')
        }
        
    except Exception as e:
        logger.error(f"Error in financial_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare-banks", response_class=EnhancedJSONResponse)
async def compare_banks(bank_names: List[str] = Query(...)):
    """Compare IT adoption across multiple banks"""
    try:
        conn = sqlite3.connect(DATABASE)
        placeholders = ','.join(['?'] * len(bank_names))
        query = f'''
            SELECT 
                bank_name,
                AVG(stock_price) as avg_stock_price,
                AVG(pe_ratio) as avg_pe_ratio,
                SUM(rnd_expenses) as total_rnd,
                AVG(it_spending) as avg_it_spending,
                SUM(digital_transactions) as total_digital,
                SUM(mobile_users) as total_mobile,
                SUM(it_news_count) as total_it_news,
                AVG(cloud_adoption_score) as avg_cloud_adoption,
                SUM(fintech_partnerships) as total_fintech_partnerships,
                SUM(api_integrations) as total_api_integrations
            FROM bank_analysis
            WHERE bank_name IN ({placeholders})
            GROUP BY bank_name
        '''
        df = pd.read_sql(query, conn, params=bank_names)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for selected banks")
            
        # Handle potential division by zero or NaN values
        for col in df.columns:
            if col != 'bank_name':
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                
        # Safely calculate normalized scores
        cols_to_normalize = [
            'avg_it_spending', 'total_digital', 'total_mobile', 
            'total_fintech_partnerships', 'total_api_integrations'
        ]
        
        # Calculate normalized scores safely (avoid division by zero)
        for col in cols_to_normalize:
            max_val = df[col].max()
            if max_val > 0:  # Avoid division by zero
                df[f"{col}_norm"] = df[col] / max_val
            else:
                df[f"{col}_norm"] = 0
        
        # Calculate IT adoption score
        df['it_adoption_score'] = (
            df['avg_it_spending_norm'] * 0.3 +
            df['total_digital_norm'] * 0.2 +
            df['total_mobile_norm'] * 0.1 +
            df['avg_cloud_adoption'] * 0.2 +
            df['total_fintech_partnerships_norm'] * 0.1 +
            df['total_api_integrations_norm'] * 0.1
        )
        
        # Clean up temporary columns
        for col in cols_to_normalize:
            df = df.drop(f"{col}_norm", axis=1)
            
        # Final check for any remaining NaN or Inf values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df.to_dict(orient='records')
        
    except Exception as e:
        logger.error(f"Error in compare_banks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/it-impact-analysis", response_class=EnhancedJSONResponse)
async def it_impact_analysis(bank_name: str):
    """Comprehensive IT impact analysis with correlations"""
    try:
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT * FROM bank_analysis
            WHERE bank_name = ?
            ORDER BY date
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Bank data not found")
            
        # Calculate correlations safely
        correlations = {}
        
        # Safe correlation function
        def safe_corr(x, y):
            if x.std() == 0 or y.std() == 0:  # Check for constant series
                return 0.0
            try:
                corr = x.corr(y)
                return 0.0 if pd.isna(corr) or np.isinf(corr) else float(corr)
            except Exception:
                return 0.0
        
        correlations = {
            "it_spending_stock": safe_corr(df['it_spending'], df['stock_price']),
            "digital_transactions_stock": safe_corr(df['digital_transactions'], df['stock_price']),
            "cloud_adoption_stock": safe_corr(df['cloud_adoption_score'], df['stock_price']),
            "it_news_sentiment_stock": safe_corr(df['it_sentiment'], df['stock_price'])
        }
        
        # Calculate growth rates safely
        def safe_pct_change(series):
            try:
                pct = series.pct_change().mean() * 100
                return 0.0 if pd.isna(pct) or np.isinf(pct) else float(pct)
            except Exception:
                return 0.0
                
        # Calculate growth metrics
        growth_metrics = {
            "stock_price_growth": safe_pct_change(df['stock_price']),
            "it_spending_growth": safe_pct_change(df['it_spending']),
            "digital_transactions_growth": safe_pct_change(df['digital_transactions']),
            "mobile_users_growth": safe_pct_change(df['mobile_users'])
        }

        # Prepare timeline data (smooth and clean it for visualization)
        timeline_df = df[['date', 'stock_price', 'it_spending', 'it_news_count', 'it_sentiment']].copy()
        timeline_df = timeline_df.fillna(method='ffill').fillna(0)

        # Generate insights based on correlations
        insights = []
        if correlations["it_spending_stock"] > 0.5:
            insights.append("Strong positive correlation between IT spending and stock performance")
        elif correlations["it_spending_stock"] > 0.3:
            insights.append("Moderate positive correlation between IT spending and stock performance")
            
        if correlations["digital_transactions_stock"] > 0.5:
            insights.append("Digital transactions strongly correlate with stock performance")
            
        if correlations["cloud_adoption_stock"] > 0.4:
            insights.append("Cloud adoption shows positive impact on stock performance")
            
        if growth_metrics["it_spending_growth"] > growth_metrics["stock_price_growth"]:
            insights.append("IT spending is growing faster than stock price, suggesting potential future returns")

        return {
            "bank": bank_name,
            "correlations": correlations,
            "growth_metrics": growth_metrics,
            "timeline": timeline_df.to_dict(orient='records'),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in it_impact_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch-fintech-news")
async def get_fintech_news(keywords: List[str] = Query(["digital banking", "fintech", "banking technology"])):
    try:
        news = fetch_fintech_news(keywords)
        return {"news": news}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

def classify_news_item(content: str) -> str:
    """Classify news based on content"""
    categories = {
        "digital_transformation": ["digital transformation", "digitalization", "digital strategy"],
        "cloud": ["cloud", "aws", "azure", "gcp"],
        "blockchain": ["blockchain", "crypto", "distributed ledger"],
        "ai_ml": ["artificial intelligence", "machine learning", "ai", "ml", "deep learning"],
        "cybersecurity": ["cybersecurity", "security", "breach", "hack", "vulnerability"],
        "mobile_banking": ["mobile banking", "app", "mobile payment"],
        "open_banking": ["open banking", "api banking", "banking-as-a-service"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in content for keyword in keywords):
            return category
            
    return "general"

@app.get("/bank-it-trends", response_class=EnhancedJSONResponse)
async def bank_it_trends(bank_name: str):
    """Get IT adoption trends for a specific bank"""
    try:
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT 
                strftime('%Y-%m', date) as month,
                AVG(it_spending) as avg_it_spending,
                SUM(digital_transactions) as monthly_digital_transactions,
                SUM(it_news_count) as monthly_it_news,
                AVG(it_sentiment) as avg_sentiment,
                MAX(cloud_adoption_score) as cloud_adoption
            FROM bank_analysis
            WHERE bank_name = ?
            GROUP BY strftime('%Y-%m', date)
            ORDER BY month
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Bank data not found")
            
        # Calculate moving averages for smoother trends
        if len(df) >= 3:  # Only calculate if we have enough data points
            df['it_spending_ma'] = df['avg_it_spending'].rolling(window=3, min_periods=1).mean()
            df['digital_trans_ma'] = df['monthly_digital_transactions'].rolling(window=3, min_periods=1).mean()
        else:
            df['it_spending_ma'] = df['avg_it_spending']
            df['digital_trans_ma'] = df['monthly_digital_transactions']
            
        # Clean NaN values
        df = df.fillna(0)
        
        # Calculate YoY growth if possible
        current_year_data = None
        prev_year_data = None
        
        if len(df) >= 12:
            current_year_data = df.iloc[-12:].copy()
            if len(df) >= 24:
                prev_year_data = df.iloc[-24:-12].copy()
        
        yoy_growth = {}
        if current_year_data is not None and prev_year_data is not None:
            for col in ['avg_it_spending', 'monthly_digital_transactions', 'cloud_adoption']:
                curr_avg = current_year_data[col].mean()
                prev_avg = prev_year_data[col].mean()
                if prev_avg > 0:  # Avoid division by zero
                    yoy_growth[f"{col}_yoy"] = ((curr_avg - prev_avg) / prev_avg) * 100
                else:
                    yoy_growth[f"{col}_yoy"] = 0.0
        
        # Generate IT adoption insights
        insights = []
        
        # Add insights based on trends
        if 'avg_it_spending_yoy' in yoy_growth:
            if yoy_growth['avg_it_spending_yoy'] > 20:
                insights.append(f"Significant increase in IT spending (up {yoy_growth['avg_it_spending_yoy']:.1f}% YoY)")
            elif yoy_growth['avg_it_spending_yoy'] < -10:
                insights.append(f"Concerning decrease in IT spending (down {abs(yoy_growth['avg_it_spending_yoy']):.1f}% YoY)")
                
        if 'monthly_digital_transactions_yoy' in yoy_growth:
            if yoy_growth['monthly_digital_transactions_yoy'] > 30:
                insights.append(f"Strong growth in digital transactions (up {yoy_growth['monthly_digital_transactions_yoy']:.1f}% YoY)")
        
        latest_cloud = float(df['cloud_adoption'].iloc[-1]) if not df.empty else 0
        if latest_cloud > 0.7:
            insights.append("High cloud adoption score indicates advanced digital transformation")
        elif latest_cloud < 0.3:
            insights.append("Low cloud adoption score suggests lagging in cloud technologies")
            
        # Calculate correlation between IT metrics and sentiment
        corr_it_sentiment = safe_corr(df['avg_it_spending'], df['avg_sentiment'])
        if abs(corr_it_sentiment) > 0.5:
            direction = "positive" if corr_it_sentiment > 0 else "negative"
            insights.append(f"Strong {direction} correlation between IT spending and news sentiment")
            
        return {
            "bank": bank_name,
            "monthly_trends": df.to_dict(orient='records'),
            "yoy_growth": yoy_growth,
            "latest_cloud_adoption": latest_cloud,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error in bank_it_trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-bank-performance")
async def analyze_bank_performance(input_data: BankDataInput):
    """Analyze bank performance based on IT metrics"""
    try:
        # Extract data from input
        bank_name = input_data.bank_name
        metrics = input_data.metrics
        
        # Create a DataFrame from the input metrics
        df = pd.DataFrame([metrics])
        df['bank_name'] = bank_name
        
        # Store in database
        store_bank_data(df)
        
        # Analyze the bank performance
        result = analyze_bank_performance(bank_name, df)
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing bank performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-stock-trend-chart")
async def create_stock_trend_chart_endpoint(req: AnalysisRequest):
    """Create stock trend chart with IT spending overlay"""
    try:
        bank_name = req.bank_name
        days = req.days
        
        # Create chart
        chart_json = create_stock_trend_chart(bank_name, days)
        
        return {"chart": chart_json}
    except Exception as e:
        logger.error(f"Error creating stock trend chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-correlation-chart")
async def create_correlation_chart_endpoint(req: AnalysisRequest):
    """Create correlation chart between IT metrics and stock performance"""
    try:
        bank_name = req.bank_name
        days = req.days
        
        # Create chart
        chart_json = create_correlation_chart(bank_name, days)
        
        return {"chart": chart_json}
    except Exception as e:
        logger.error(f"Error creating correlation chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-json-data")
async def process_json_data(data: dict):
    """Process custom JSON data for analysis"""
    try:
        # Process the data
        result = handle_json(data)
        
        return result
    except Exception as e:
        logger.error(f"Error processing JSON data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk-metrics/{bank_name}", response_class=EnhancedJSONResponse)
async def get_risk_metrics(bank_name: str):
    """Get financial risk metrics for a bank"""
    try:
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT date, stock_price FROM bank_analysis
            WHERE bank_name = ?
            ORDER BY date
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Bank data not found")
            
        # Calculate returns
        df['returns'] = np.log(df['stock_price'] / df['stock_price'].shift(1))
        returns = df['returns'].dropna()
        
        if len(returns) < 2:  # Need at least 2 data points
            raise HTTPException(status_code=400, detail="Insufficient data for risk calculation")
        
        # Calculate metrics safely
        risk_metrics = {
            "daily_volatility": float(returns.std()),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "value_at_risk_95": float(norm.ppf(0.05, returns.mean(), returns.std())),
            "conditional_var_95": float(returns[returns <= norm.ppf(0.05, returns.mean(), returns.std())].mean()),
            "returns_skewness": float(returns.skew()),
            "returns_kurtosis": float(returns.kurtosis()),
            "sharpe_ratio": float((returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0.0,
            "sortino_ratio": float((returns.mean() * 252 - RISK_FREE_RATE) / (returns[returns < 0].std() * np.sqrt(252))) if returns[returns < 0].std() > 0 else 0.0
        }
        
        return {
            "bank": bank_name,
            "risk_metrics": risk_metrics
        }
        
    except Exception as e:
        logger.error(f"Error in risk_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/monte-carlo-forecast/{bank_name}", response_class=EnhancedJSONResponse)
async def monte_carlo_forecast(bank_name: str, days: int = 30):
    """Generate Monte Carlo forecast for stock prices"""
    try:
        conn = sqlite3.connect(DATABASE)
        query = '''
            SELECT stock_price FROM bank_analysis
            WHERE bank_name = ?
            ORDER BY date DESC
            LIMIT 252 
        '''
        df = pd.read_sql(query, conn, params=[bank_name])
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Bank data not found")
            
        returns = np.log(df['stock_price'] / df['stock_price'].shift(1)).dropna()
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        
        # Generate forecasts
        S0 = df['stock_price'].iloc[-1]
        paths = FinancialModel.gbm_model(S0, mu, sigma, days, MONTE_CARLO_SIMULATIONS)
        
        return {
            "bank": bank_name,
            "forecast_days": days,
            "current_price": float(S0),
            "expected_return": float(mu),
            "volatility": float(sigma),
            "percentiles": {
                "5th": float(np.percentile(paths[-1], 5)),
                "50th": float(np.percentile(paths[-1], 50)),
                "95th": float(np.percentile(paths[-1], 95))
            },
            "paths": paths.tolist()
        }
    except Exception as e:
        logger.error(f"Error in monte_carlo_forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)