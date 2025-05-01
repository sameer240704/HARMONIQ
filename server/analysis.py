import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

def analyze_bank_performance(df: pd.DataFrame, metrics: List[str]) -> Dict:
    """Comprehensive analysis of bank performance data"""
    analysis = {}
    
    # Add basic statistics
    analysis['basic_stats'] = df.groupby('bank_name')[metrics].agg(['mean', 'median', 'std', 'min', 'max'])
    
    # Add trend analysis
    analysis['trends'] = {}
    for metric in metrics:
        analysis['trends'][metric] = {
            '30_day_ma': df.groupby('date')[metric].mean().rolling(30).mean().to_dict(),
            'growth_rate': df.groupby('bank_name')[metric].apply(
                lambda x: x.pct_change().mean() * 100
            ).to_dict()
        }
    
    # Add correlation matrix
    analysis['correlations'] = df[metrics].corr().to_dict()
    
    # Add performance comparison
    analysis['comparison'] = {}
    for metric in metrics:
        analysis['comparison'][metric] = {
            'best_performer': df.groupby('bank_name')[metric].mean().idxmax(),
            'worst_performer': df.groupby('bank_name')[metric].mean().idxmin(),
            'industry_average': df[metric].mean()
        }
    
    # Generate visualization data
    analysis['visualizations'] = {
        'stock_price_trend': create_stock_trend_chart(df).to_json(),
        'it_spending_correlation': create_correlation_chart(df).to_json()
    }
    
    return analysis

def create_stock_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive stock price trend chart"""
    fig = go.Figure()
    
    for bank in df['bank_name'].unique():
        bank_df = df[df['bank_name'] == bank]
        fig.add_trace(go.Scatter(
            x=bank_df['date'],
            y=bank_df['stock_price'],
            name=bank,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Stock Price Trends',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        hovermode='x unified'
    )
    return fig

def create_correlation_chart(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    corr = df[['stock_price', 'it_news_count', 'rnd_expenses', 'digital_mentions']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='Metrics Correlation Matrix',
        xaxis_title='Metrics',
        yaxis_title='Metrics'
    )
    return fig

def handle_json(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")