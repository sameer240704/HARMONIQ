import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime

def analyze_bank_performance(df: pd.DataFrame, metrics: List[str]) -> Dict:
    """Comprehensive analysis of bank performance data"""
    analysis = {}
    
    # Ensure numeric columns are properly typed
    for metric in metrics:
        if metric in df.columns:
            if metric not in ['bank_name', 'date']:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Add basic statistics
    analysis['basic_stats'] = {}
    grouped = df.groupby('bank_name')
    for metric in metrics:
        if metric not in ['bank_name', 'date']:
            analysis['basic_stats'][metric] = {
                bank: {
                    'mean': float(group[metric].mean()),
                    'median': float(group[metric].median()),
                    'std': float(group[metric].std()),
                    'min': float(group[metric].min()),
                    'max': float(group[metric].max())
                }
                for bank, group in grouped
            }
    
    # Add trend analysis
    analysis['trends'] = {}
    for metric in metrics:
        if metric not in ['bank_name', 'date']:
            analysis['trends'][metric] = {
                '30_day_ma': {
                    date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date: float(value)
                    for date, value in df.groupby('date')[metric].mean().rolling(30).mean().dropna().items()
                },
                'growth_rate': {
                    bank: float(group[metric].pct_change().mean() * 100)
                    for bank, group in grouped
                }
            }
    
    # Add correlation matrix
    numeric_metrics = [m for m in metrics if m not in ['bank_name', 'date']]
    if len(numeric_metrics) > 1:
        corr_matrix = df[numeric_metrics].corr().fillna(0)
        analysis['correlations'] = {
            col1: {col2: float(corr_matrix.loc[col1, col2]) for col2 in corr_matrix.columns}
            for col1 in corr_matrix.index
        }
    
    # Add performance comparison
    analysis['comparison'] = {}
    for metric in metrics:
        if metric not in ['bank_name', 'date']:
            bank_means = df.groupby('bank_name')[metric].mean()
            analysis['comparison'][metric] = {
                'best_performer': bank_means.idxmax(),
                'worst_performer': bank_means.idxmin(),
                'industry_average': float(df[metric].mean())
            }
    
    # Generate visualization data
    analysis['visualizations'] = {
        'stock_price_trend': create_stock_trend_chart(df),
        'it_spending_correlation': create_correlation_chart(df)
    }
    
    return analysis

def create_stock_trend_chart(df: pd.DataFrame) -> dict:
    """Create interactive stock price trend chart data"""
    chart_data = []
    
    for bank in df['bank_name'].unique():
        bank_df = df[df['bank_name'] == bank]
        chart_data.append({
            'bank': bank,
            'dates': [date for date in bank_df['date']],
            'prices': [float(price) for price in bank_df['stock_price']]
        })
    
    return {
        'title': 'Stock Price Trends',
        'series': chart_data
    }

def create_correlation_chart(df: pd.DataFrame) -> dict:
    """Create correlation heatmap data"""
    metrics = ['stock_price', 'it_news_count', 'rnd_expenses', 'digital_mentions']
    metrics = [m for m in metrics if m in df.columns]
    
    if len(metrics) < 2:
        return {'error': 'Not enough metrics for correlation'}
        
    # Ensure numeric columns
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    corr = df[metrics].corr().fillna(0)
    
    return {
        'title': 'Metrics Correlation Matrix',
        'labels': list(corr.columns),
        'values': [
            [float(corr.iloc[i, j]) for j in range(len(corr.columns))]
            for i in range(len(corr.index))
        ]
    }

def handle_json(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")