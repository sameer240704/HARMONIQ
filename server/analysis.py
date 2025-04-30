import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from prophet import Prophet

def analyze_bank_performance(df, metrics):
    """
    Analyze bank performance based on selected metrics
    """
    results = {}
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Group by bank and calculate metrics
    grouped = df.groupby('bank_name')
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        metric_results = {}
        
        for bank, data in grouped:
            # Basic statistics
            metric_data = data[metric]
            metric_results[bank] = {
                'mean': float(metric_data.mean()),
                'median': float(metric_data.median()),
                'std_dev': float(metric_data.std()),
                'min': float(metric_data.min()),
                'max': float(metric_data.max()),
                'growth_rate': float(calculate_growth_rate(metric_data)) if calculate_growth_rate(metric_data) is not None else None,
                'correlation_with_it': float(calculate_correlation(data, metric, 'it_spending')) if calculate_correlation(data, metric, 'it_spending') is not None else None
            }
        
        results[metric] = metric_results
    
    # Calculate IT efficiency metrics
    it_efficiency = {}
    for bank, data in grouped:
        try:
            efficiency = (data['digital_transactions'].mean() / data['it_spending'].mean()) * 100
            it_efficiency[bank] = float(efficiency)
        except:
            it_efficiency[bank] = None
    
    results['it_efficiency'] = it_efficiency
    
    return results

def calculate_growth_rate(series):
    """
    Calculate compound annual growth rate (CAGR)
    """
    if len(series) < 2:
        return None
    start = series.iloc[0]
    end = series.iloc[-1]
    periods = len(series)
    if start <= 0:
        return None
    return (end / start) ** (1/periods) - 1

def calculate_correlation(data, metric1, metric2):
    """
    Calculate correlation between two metrics
    """
    if metric1 not in data.columns or metric2 not in data.columns:
        return None
    return data[[metric1, metric2]].corr().iloc[0,1]

def predict_future_trends(df, periods=365):
    """
    Predict future trends using Facebook Prophet
    """
    # Prepare data for Prophet
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Forecast
    forecast = model.predict(future)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        name='Actual',
        mode='markers'
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='royalblue')
    ))
    
    # Add uncertainty interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='Uncertainty'
    ))
    
    # Convert figure to dict
    plot_json = fig.to_dict()
    
    # Prepare forecast data
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_data = forecast_data.rename(columns={
        'ds': 'date',
        'yhat': 'prediction',
        'yhat_lower': 'lower_bound',
        'yhat_upper': 'upper_bound'
    })
    
    # Convert datetime to string to make it JSON serializable
    forecast_data['date'] = forecast_data['date'].dt.strftime('%Y-%m-%d')
    
    # Create a simplified components dict since model.plot_components() returns a figure
    components = {
        'trend': forecast[['ds', 'trend']].tail(periods).to_dict('records'),
        'yearly': forecast[['ds', 'yearly']].tail(periods).to_dict('records') if 'yearly' in forecast.columns else None,
        'weekly': forecast[['ds', 'weekly']].tail(periods).to_dict('records') if 'weekly' in forecast.columns else None
    }
    
    return {
        'plot': plot_json,
        'forecast': forecast_data.to_dict('records'),
        'trend_components': components
    }

def calculate_digital_adoption_metrics(df):
    """
    Calculate digital adoption metrics and trends
    """
    results = {}
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate growth rates
    for metric in ['digital_transactions', 'mobile_users', 'it_spending']:
        if metric not in df.columns:
            continue
            
        # Calculate monthly growth rate
        df[f'{metric}_growth'] = df[metric].pct_change() * 100
        
        # Fit linear regression to estimate trend
        X = (df['date'] - df['date'].min()).dt.days.values.reshape(-1, 1)
        y = df[metric].values
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        model = LinearRegression().fit(X, y)
        trend = model.predict(X)
        r2 = r2_score(y, trend)
        
        # Convert datetime to string for JSON serialization
        dates_str = df['date'].dt.strftime('%Y-%m-%d').tolist()
        
        results[metric] = {
            'current_value': float(df[metric].iloc[-1]),
            'growth_rate': float(df[f'{metric}_growth'].iloc[-1]) if not pd.isna(df[f'{metric}_growth'].iloc[-1]) else None,
            'avg_growth_rate': float(df[f'{metric}_growth'].mean()) if not pd.isna(df[f'{metric}_growth'].mean()) else None,
            'trend_strength': float(r2),
            'historical_data': [{'date': date, metric: float(value)} for date, value in zip(dates_str, df[metric].tolist())]
        }
    
    # Calculate adoption efficiency (transactions per IT dollar)
    if 'digital_transactions' in df.columns and 'it_spending' in df.columns:
        df['adoption_efficiency'] = df['digital_transactions'] / df['it_spending']
        dates_str = df['date'].dt.strftime('%Y-%m-%d').tolist()
        results['adoption_efficiency'] = {
            'current_value': float(df['adoption_efficiency'].iloc[-1]),
            'historical_data': [{'date': date, 'adoption_efficiency': float(value)} for date, value in zip(dates_str, df['adoption_efficiency'].tolist())]
        }
    
    return results

def compare_banks_technology_spending(df):
    """
    Compare technology spending and outcomes across banks
    """
    results = {}
    
    # Calculate averages
    avg_metrics = df.groupby('bank_name')[['avg_it_spending', 'avg_digital_transactions', 'avg_mobile_users']].mean()
    
    # Calculate ROI metrics
    avg_metrics['transactions_per_it_dollar'] = avg_metrics['avg_digital_transactions'] / avg_metrics['avg_it_spending']
    avg_metrics['users_per_it_dollar'] = avg_metrics['avg_mobile_users'] / avg_metrics['avg_it_spending']
    
    # Convert to dict and ensure all values are JSON serializable
    avg_metrics_dict = {}
    for bank in avg_metrics.index:
        avg_metrics_dict[bank] = {k: float(v) for k, v in avg_metrics.loc[bank].to_dict().items()}
    
    results['average_metrics'] = avg_metrics_dict
    
    # Prepare data for visualizations (simplified since correlation cannot be calculated with just average values)
    visualization_data = {
        'it_spending': [{'bank_name': bank, 'value': float(value)} for bank, value in zip(df['bank_name'], df['avg_it_spending'])],
        'digital_transactions': [{'bank_name': bank, 'value': float(value)} for bank, value in zip(df['bank_name'], df['avg_digital_transactions'])],
        'mobile_users': [{'bank_name': bank, 'value': float(value)} for bank, value in zip(df['bank_name'], df['avg_mobile_users'])]
    }
    
    results['visualization_data'] = visualization_data
    
    return results