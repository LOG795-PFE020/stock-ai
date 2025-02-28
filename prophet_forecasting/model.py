"""
Module for training and evaluating the Prophet model.
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import pickle
from prophet import Prophet
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(
    changepoint_prior_scale: float = config.DEFAULT_CHANGEPOINT_PRIOR_SCALE,
    seasonality_mode: str = config.DEFAULT_SEASONALITY_MODE,
    daily_seasonality: bool = config.DEFAULT_DAILY_SEASONALITY,
    weekly_seasonality: bool = config.DEFAULT_WEEKLY_SEASONALITY,
    yearly_seasonality: bool = config.DEFAULT_YEARLY_SEASONALITY,
    seasonality_prior_scale: float = config.DEFAULT_SEASONALITY_PRIOR_SCALE,
    holidays_prior_scale: float = config.DEFAULT_HOLIDAYS_PRIOR_SCALE,
    changepoint_range: float = config.DEFAULT_CHANGEPOINT_RANGE
) -> Prophet:
    """
    Create a new Prophet model with the specified parameters.
    
    Args:
        changepoint_prior_scale: Controls flexibility of the trend
        seasonality_mode: 'additive' or 'multiplicative'
        daily_seasonality: Whether to include daily seasonality
        weekly_seasonality: Whether to include weekly seasonality
        yearly_seasonality: Whether to include yearly seasonality
        seasonality_prior_scale: Controls flexibility of the seasonality
        holidays_prior_scale: Controls flexibility of the holiday effects
        changepoint_range: Proportion of history in which trend changepoints are placed
        
    Returns:
        Configured Prophet model
    """
    # Initialize Prophet with advanced configuration
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        changepoint_range=changepoint_range,
        interval_width=0.95  # 95% prediction intervals
    )
    
    # Add US holidays for better stock market modeling
    try:
        from prophet.holidays import make_holidays_df
        import pandas as pd
        
        # Create a dataframe of US holidays for 6 years (including future dates)
        years = list(range(dt.datetime.now().year - 5, dt.datetime.now().year + 2))
        holidays = pd.DataFrame(make_holidays_df(years=years, country='US'))
        
        # Stock-market specific holidays and events
        custom_holidays = pd.DataFrame({
            'holiday': 'market_close',
            'ds': pd.to_datetime([
                # Add additional trading holidays or significant market events
                '2020-03-16',  # COVID-19 market crash
                '2021-01-27',  # GameStop short squeeze
                # Add more market significant dates as needed
            ]),
            'lower_window': 0,
            'upper_window': 1,
        })
        
        # Combine holidays
        all_holidays = pd.concat([holidays, custom_holidays])
        model.add_country_holidays(country_name='US')
        
        logger.info("Added US holidays to model")
    except Exception as e:
        logger.warning(f"Could not add holiday effects: {e}")
    
    # Add custom seasonalities that might be relevant for stocks
    try:
        # Monthly seasonality (e.g. beginning vs end of month effects)
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5,
            mode=seasonality_mode
        )
        
        # Quarterly seasonality (for earnings impacts)
        model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=5,
            mode=seasonality_mode
        )
        
        logger.info("Added custom seasonalities to model")
    except Exception as e:
        logger.warning(f"Could not add custom seasonalities: {e}")
    
    return model

def train_model(
    data: pd.DataFrame, 
    model_params: Optional[Dict[str, Any]] = None,
    add_regressors: bool = True
) -> Prophet:
    """
    Train a Prophet model on the provided data.
    
    Args:
        data: DataFrame with 'ds' and 'y' columns
        model_params: Dictionary of model parameters
        add_regressors: Whether to automatically add additional columns as regressors
        
    Returns:
        Trained Prophet model
    """
    # Use default params if none provided
    if model_params is None:
        model_params = {}
    
    # Create model with specified params
    model = create_model(**model_params)
    
    # Identify any regressor columns (all columns except 'ds' and 'y')
    if add_regressors:
        regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]
        
        if regressor_columns:
            logger.info(f"Adding {len(regressor_columns)} regressors to model: {regressor_columns}")
            
            # Add each regressor to the model
            for regressor in regressor_columns:
                # If the regressor name contains certain keywords, use a higher prior scale
                if any(keyword in regressor.lower() for keyword in ['momentum', 'volatility', 'volume']):
                    try:
                        model.add_regressor(regressor, prior_scale=10.0, 
                                            mode=model_params.get('seasonality_mode', 'multiplicative'))
                        logger.info(f"Added market indicator regressor: {regressor} with higher prior scale")
                    except Exception as e:
                        logger.warning(f"Could not add regressor {regressor}: {e}")
                else:
                    try:
                        model.add_regressor(regressor)
                        logger.info(f"Added regressor: {regressor}")
                    except Exception as e:
                        logger.warning(f"Could not add regressor {regressor}: {e}")
    
    # Check if data has any NaN values and drop them
    if data.isna().any().any():
        original_len = len(data)
        data = data.dropna()
        logger.warning(f"Dropped {original_len - len(data)} rows with NaN values")
    
    # Fit model to data
    logger.info(f"Training model on {len(data)} data points")
    model.fit(data)
    
    return model

def evaluate_model(
    model: Prophet, 
    data: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate a trained model on the provided data.
    
    Args:
        model: Trained Prophet model
        data: DataFrame with 'ds' and 'y' columns to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Check for regressors in the model
    regressor_names = [reg for reg in model.extra_regressors.keys()] if hasattr(model, 'extra_regressors') else []
    
    # Create a dataframe for prediction that has all required columns
    predict_df = data.copy()
    
    # For any missing regressors, add dummy values (zeros)
    for reg in regressor_names:
        if reg not in predict_df.columns:
            logger.warning(f"Regressor '{reg}' missing from evaluation data. Using zeros.")
            predict_df[reg] = 0.0
    
    # Make predictions 
    try:
        forecast = model.predict(predict_df)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Try to make a simpler prediction with just the dates
        forecast = model.predict(predict_df[['ds']])
    
    # Extract actual and predicted values
    y_true = data['y'].values
    y_pred = forecast['yhat'].values
    
    # Calculate basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE (Mean Absolute Percentage Error) with protection against division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((y_true - y_pred) / y_true) * 100
        mape = np.nanmean(mape_values[~np.isinf(mape_values)])
    
    # Calculate R² (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate median absolute error
    med_ae = median_absolute_error(y_true, y_pred)
    
    # Calculate max error
    max_err = max_error(y_true, y_pred)
    
    # Determine directional accuracy (percentage of correct up/down predictions)
    if len(y_true) > 1 and len(y_pred) > 1:
        y_true_dir = np.sign(np.diff(y_true))
        y_pred_dir = np.sign(np.diff(y_pred))
        # Protect against having all zeros in the direction arrays
        if np.count_nonzero(y_true_dir) > 0 and np.count_nonzero(y_pred_dir) > 0:
            directional_accuracy = np.mean(y_true_dir == y_pred_dir) * 100
        else:
            directional_accuracy = 50.0  # Default to coin flip if no directional changes
    else:
        directional_accuracy = 50.0
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_absolute_error': med_ae,
        'max_error': max_err,
        'directional_accuracy': directional_accuracy
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    
    return metrics

def tune_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    param_grid: Optional[Dict[str, list]] = None,
    use_regressors: bool = True
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune model hyperparameters using the validation set.
    
    Args:
        train_data: Training data
        val_data: Validation data
        param_grid: Dictionary of parameters to tune
        use_regressors: Whether to use additional regressor columns in the data
        
    Returns:
        Tuple of (best_params, best_metrics)
    """
    # Default parameter grid if none provided - expanded with more options
    if param_grid is None:
        # For financial data, especially stocks, multiplicative seasonality often works better
        # and we want to focus on the changepoint parameters
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1],
            'seasonality_mode': ['multiplicative'],  # Stocks usually have multiplicative seasonality
            'changepoint_range': [0.85, 0.9, 0.95]   # Look at more recent changepoints
        }
    
    # Initialize tracking of best model
    best_metrics = {'mae': float('inf')}
    best_params = {}
    
    # Log the tuning process
    logger.info(f"Tuning model with parameter grid: {param_grid}")
    
    # For efficiency, we'll use a two-stage approach:
    # 1. First, tune the most impactful parameters: changepoint_prior_scale and seasonality_mode
    # 2. Then fine-tune using the other parameters
    
    # Stage 1: Basic tuning
    stage1_grid = {
        'changepoint_prior_scale': param_grid['changepoint_prior_scale'],
        'seasonality_mode': param_grid['seasonality_mode']
    }
    
    # Generate parameter combinations for stage 1
    from itertools import product
    stage1_keys = list(stage1_grid.keys())
    stage1_values = list(stage1_grid.values())
    
    # Loop through stage 1 combinations
    logger.info("Stage 1: Tuning changepoint_prior_scale and seasonality_mode")
    for combination in product(*stage1_values):
        current_params = dict(zip(stage1_keys, combination))
        
        logger.info(f"Testing parameters: {current_params}")
        
        # Train model with current parameters
        model = train_model(train_data, current_params, add_regressors=use_regressors)
        
        # Evaluate on validation data
        metrics = evaluate_model(model, val_data)
        
        # Check if this is the best model so far
        if metrics['mae'] < best_metrics['mae']:
            best_metrics = metrics
            best_params = current_params
            logger.info(f"New best model: {best_params}, MAE: {best_metrics['mae']}")
    
    # Stage 2: Fine-tuning with the remaining parameters
    # But only if we have additional parameters to tune
    if len(param_grid) > 2:
        logger.info(f"Stage 2: Fine-tuning with best parameters from stage 1: {best_params}")
        
        # Build the stage 2 grid based on the best params from stage 1
        stage2_grid = {k: v for k, v in param_grid.items() if k not in best_params}
        
        # If there are parameters to tune in stage 2
        if stage2_grid:
            stage2_keys = list(stage2_grid.keys())
            stage2_values = list(stage2_grid.values())
            
            # Loop through stage 2 combinations
            for combination in product(*stage2_values):
                current_params = dict(zip(stage2_keys, combination))
                
                # Combine with best params from stage 1
                current_params.update(best_params)
                
                logger.info(f"Testing parameters: {current_params}")
                
                # Train model with current parameters
                model = train_model(train_data, current_params, add_regressors=use_regressors)
                
                # Evaluate on validation data
                metrics = evaluate_model(model, val_data)
                
                # Check if this is the best model so far
                if metrics['mae'] < best_metrics['mae']:
                    best_metrics = metrics
                    best_params = current_params
                    logger.info(f"New best model: {best_params}, MAE: {best_metrics['mae']}")
    
    logger.info(f"Best model parameters: {best_params}")
    logger.info(f"Best model metrics: {best_metrics}")
    
    # Create a simplified parameter set with only the parameters that differ from default
    default_params = {
        'changepoint_prior_scale': config.DEFAULT_CHANGEPOINT_PRIOR_SCALE,
        'seasonality_mode': config.DEFAULT_SEASONALITY_MODE,
        'seasonality_prior_scale': config.DEFAULT_SEASONALITY_PRIOR_SCALE,
        'holidays_prior_scale': config.DEFAULT_HOLIDAYS_PRIOR_SCALE,
        'changepoint_range': config.DEFAULT_CHANGEPOINT_RANGE
    }
    
    # Only include parameters that differ from default
    simplified_params = {k: v for k, v in best_params.items() if k not in default_params or v != default_params[k]}
    
    # If no parameters are better than default, include at least changepoint_prior_scale
    if not simplified_params and 'changepoint_prior_scale' in best_params:
        simplified_params['changepoint_prior_scale'] = best_params['changepoint_prior_scale']
    
    logger.info(f"Simplified best parameters: {simplified_params}")
    
    return simplified_params, best_metrics

def generate_forecast(
    model: Prophet, 
    data: pd.DataFrame,
    periods: int = config.FORECAST_DAYS
) -> pd.DataFrame:
    """
    Generate forecast for the specified number of periods.
    
    Args:
        model: Trained Prophet model
        data: Historical data used for training (with additional regressors)
        periods: Number of days to forecast
        
    Returns:
        DataFrame with forecast
    """
    logger.info(f"Generating {periods} day forecast")
    
    # Identify any regressor columns (all columns except 'ds' and 'y')
    regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]
    
    if regressor_columns:
        logger.info(f"Model has {len(regressor_columns)} additional regressors: {regressor_columns}")
        
        # Create future dataframe with regressor values
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # For each regressor, we need to provide future values
        # Strategy: use the most recent values from historical data for future predictions
        future_regressors = {}
        
        for col in regressor_columns:
            # Get the most recent 30 days of values
            recent_values = data[col].iloc[-30:].values
            
            # Calculate the median of recent values to use for future
            if np.issubdtype(recent_values.dtype, np.number):  # Check if numeric
                future_val = np.median(recent_values[~np.isnan(recent_values)])
                
                # For day-specific columns, we need to compute them correctly
                if col == 'day_of_week':
                    # Calculate proper day of week for future dates
                    future[col] = future['ds'].dt.dayofweek
                elif col == 'month':
                    # Calculate proper month for future dates
                    future[col] = future['ds'].dt.month
                elif col == 'month_end':
                    # Check if date is month end
                    future[col] = future['ds'].dt.is_month_end.astype(int)
                elif col == 'month_start':
                    # Check if date is start of month (first 3 days)
                    next_month = future['ds'] + pd.Timedelta(days=3)
                    future[col] = (future['ds'].dt.month != next_month.dt.month).astype(int)
                else:
                    # For other regressors, use median of recent values
                    future.loc[future['ds'] > data['ds'].max(), col] = future_val
                    
                    # For historical dates, use actual historical values
                    for date in future.loc[future['ds'] <= data['ds'].max(), 'ds']:
                        mask = data['ds'] == date
                        if mask.any():
                            future.loc[future['ds'] == date, col] = data.loc[mask, col].values[0]
            
            else:  # For non-numeric columns, just use the most frequent value
                future_val = data[col].value_counts().index[0]
                future.loc[future['ds'] > data['ds'].max(), col] = future_val
                
                # For historical dates, use actual historical values
                for date in future.loc[future['ds'] <= data['ds'].max(), 'ds']:
                    mask = data['ds'] == date
                    if mask.any():
                        future.loc[future['ds'] == date, col] = data.loc[mask, col].values[0]
    else:
        # If no regressors, just create a simple future dataframe
        future = model.make_future_dataframe(periods=periods)
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast

def get_specific_forecasts(
    forecast: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract specific forecast points (24h, 48h, 1w).
    
    Args:
        forecast: DataFrame with Prophet forecast
        
    Returns:
        Dictionary with specific forecasts
    """
    logger.info("Extracting specific forecast points")
    
    try:
        # Get the last date in the historical data
        now = pd.Timestamp.now()
        historical_data = forecast[forecast['ds'] <= now]
        
        if historical_data.empty:
            logger.warning("No historical data found in forecast. Using current date.")
            last_historical_date = now
        else:
            last_historical_date = historical_data['ds'].max()
            
        logger.info(f"Last historical date: {last_historical_date}")
        
        # Get forecast dates
        forecast_24h = last_historical_date + pd.Timedelta(days=1)
        forecast_48h = last_historical_date + pd.Timedelta(days=2)
        forecast_1w = last_historical_date + pd.Timedelta(days=7)
        
        logger.info(f"Forecast dates: 24h={forecast_24h.date()}, 48h={forecast_48h.date()}, 1w={forecast_1w.date()}")
        
        # Find the closest dates in case exact matches aren't found
        def find_closest_date(target_date):
            # Calculate absolute difference between target date and all forecast dates
            forecast['date_diff'] = abs(forecast['ds'] - target_date)
            # Get the row with minimum difference
            closest_row = forecast.loc[forecast['date_diff'].idxmin()]
            logger.info(f"Closest date to {target_date.date()} is {closest_row['ds'].date()}")
            return closest_row['yhat']
        
        # Get predicted values for these dates (with fallback to closest date)
        try:
            forecast_24h_value = forecast[forecast['ds'] == forecast_24h]['yhat'].iloc[0]
        except (IndexError, KeyError):
            logger.warning(f"Exact match for 24h forecast date not found, using closest date")
            forecast_24h_value = find_closest_date(forecast_24h)
            
        try:
            forecast_48h_value = forecast[forecast['ds'] == forecast_48h]['yhat'].iloc[0]
        except (IndexError, KeyError):
            logger.warning(f"Exact match for 48h forecast date not found, using closest date")
            forecast_48h_value = find_closest_date(forecast_48h)
            
        try:
            forecast_1w_value = forecast[forecast['ds'] == forecast_1w]['yhat'].iloc[0]
        except (IndexError, KeyError):
            logger.warning(f"Exact match for 1w forecast date not found, using closest date")
            forecast_1w_value = find_closest_date(forecast_1w)
        
        # Round to 2 decimal places
        return {
            '24h': round(float(forecast_24h_value), 2),
            '48h': round(float(forecast_48h_value), 2),
            '1w': round(float(forecast_1w_value), 2)
        }
    except Exception as e:
        logger.error(f"Error extracting specific forecasts: {e}")
        # Return placeholder values in case of error
        return {
            '24h': 0.0,
            '48h': 0.0,
            '1w': 0.0
        }

def get_recent_performance(
    model: Prophet,
    data: pd.DataFrame,
    days: int = 5
) -> Dict[str, Any]:
    """
    Evaluate model performance on the most recent days of historical data.
    
    Args:
        model: Trained Prophet model
        data: Complete historical data
        days: Number of recent days to analyze
        
    Returns:
        Dictionary with recent performance details
    """
    # Get the most recent days of data
    if len(data) <= days:
        recent_data = data.copy()
    else:
        recent_data = data.tail(days).copy()
    
    # Check for regressors in the model
    regressor_names = [reg for reg in model.extra_regressors.keys()] if hasattr(model, 'extra_regressors') else []
    
    # Create a dataframe for prediction that has all required columns
    predict_df = recent_data.copy()
    
    # For any missing regressors, add dummy values (zeros)
    for reg in regressor_names:
        if reg not in predict_df.columns:
            logger.warning(f"Regressor '{reg}' missing from evaluation data. Using zeros.")
            predict_df[reg] = 0.0
            
    # Make predictions
    try:
        predictions = model.predict(predict_df)
    except Exception as e:
        logger.error(f"Error making recent predictions: {e}")
        # Try to make a simpler prediction with just the dates
        predictions = model.predict(predict_df[['ds']])
    
    # Create a detailed comparison
    comparison = []
    
    for i, (_, row) in enumerate(recent_data.iterrows()):
        actual = row['y']
        date = row['ds']
        
        # Find the corresponding prediction
        pred_row = predictions[predictions['ds'] == date]
        if not pred_row.empty:
            predicted = pred_row['yhat'].iloc[0]
            lower = pred_row['yhat_lower'].iloc[0]
            upper = pred_row['yhat_upper'].iloc[0]
            
            # Calculate error
            error = actual - predicted
            error_pct = (error / actual) * 100
            
            # Determine if actual is within prediction interval
            within_interval = lower <= actual <= upper
            
            comparison.append({
                'date': date.strftime('%Y-%m-%d'),
                'actual': round(float(actual), 2),
                'predicted': round(float(predicted), 2),
                'error': round(float(error), 2),
                'error_percent': round(float(error_pct), 2),
                'lower_bound': round(float(lower), 2),
                'upper_bound': round(float(upper), 2),
                'within_interval': within_interval
            })
    
    # Calculate overall metrics for recent data
    recent_metrics = evaluate_model(model, recent_data)
    
    # Calculate percentage of predictions within interval
    within_interval_pct = sum(c['within_interval'] for c in comparison) / len(comparison) * 100
    
    return {
        'days': days,
        'comparison': comparison,
        'metrics': recent_metrics,
        'within_interval_percentage': within_interval_pct
    }

def save_model(model: Prophet, ticker: str) -> str:
    """
    Save a trained Prophet model to disk.
    
    Args:
        model: Trained Prophet model to save
        ticker: Ticker symbol for naming the saved model
        
    Returns:
        Path to the saved model file
    """
    # Ensure the models directory exists
    os.makedirs(config.DATA_MODELS_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d")
    filename = f"{ticker.lower()}_model_{timestamp}.pkl"
    filepath = os.path.join(config.DATA_MODELS_DIR, filename)
    
    # Save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")
    return filepath

def load_model(filepath: str) -> Prophet:
    """
    Load a saved Prophet model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Loaded Prophet model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model

def get_latest_model(ticker: str) -> Optional[str]:
    """
    Find the latest saved model for a given ticker.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Path to the latest model file, or None if no model is found
    """
    if not os.path.exists(config.DATA_MODELS_DIR):
        return None
    
    # Find all model files for this ticker
    ticker_prefix = f"{ticker.lower()}_model_"
    model_files = [f for f in os.listdir(config.DATA_MODELS_DIR) 
                   if f.startswith(ticker_prefix) and f.endswith('.pkl')]
    
    if not model_files:
        return None
    
    # Sort by timestamp (which is part of the filename)
    latest_model = sorted(model_files)[-1]
    return os.path.join(config.DATA_MODELS_DIR, latest_model)

def train_and_forecast(
    ticker: str,
    data: Dict[str, pd.DataFrame],
    tune: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
    quick_mode: bool = False,
    save: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline to train a model and generate forecasts.
    
    Args:
        ticker: Stock ticker symbol
        data: Dictionary with 'train', 'val', 'test', and 'train_val' dataframes
        tune: Whether to tune the model hyperparameters
        model_params: Model parameters to use (if tune is False)
        quick_mode: If True, use simplified tuning for faster results
        save: Whether to save the trained model to disk
        
    Returns:
        Dictionary with model, metrics, and forecasts
    """
    # Check what regressors are available
    available_regressors = [col for col in data['train'].columns if col not in ['ds', 'y']]
    
    # Log what regressors are being used
    if available_regressors:
        logger.info(f"Using {len(available_regressors)} additional regressors: {available_regressors}")
    
    # Configure initial training to disable automatic regressor addition for tuning
    # since we'll handle it manually for each model
    if tune:
        # Create modified training data with additional regressor columns for tuning
        train_data_with_regressors = data['train'].copy()
        val_data_with_regressors = data['val'].copy()
        
        if quick_mode:
            # Use a simplified parameter grid for faster tuning
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.05, 0.1],
                'seasonality_mode': ['multiplicative']
            }
            # Tune model with regressors
            best_params, val_metrics = tune_model(train_data_with_regressors, val_data_with_regressors, 
                                               param_grid, use_regressors=True)
        else:
            # Full tuning with regressors
            best_params, val_metrics = tune_model(train_data_with_regressors, val_data_with_regressors, 
                                               use_regressors=True)
        
        logger.info(f"Tuned model parameters: {best_params}")
    else:
        # Use provided or default parameters
        best_params = model_params or {}
        val_metrics = {}
    
    # Train final model on combined train+validation data WITH regressors
    final_model = train_model(data['train_val'], best_params, add_regressors=True)
    
    # Evaluate on test data
    test_metrics = evaluate_model(final_model, data['test'])
    
    # Generate forecast using the full dataset to ensure consistency with regressors
    forecast = generate_forecast(final_model, data['full'])
    
    # Extract specific forecasts
    specific_forecasts = get_specific_forecasts(forecast)
    
    # Get recent performance analysis (last 5 days)
    recent_performance = get_recent_performance(final_model, data['full'], days=5)
    
    # Save the model if requested
    model_path = None
    if save:
        try:
            model_path = save_model(final_model, ticker)
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
    
    # Return results
    return {
        'model': final_model,
        'forecast': forecast,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'specific_forecasts': specific_forecasts,
        'params': best_params,
        'recent_performance': recent_performance,
        'model_path': model_path
    }