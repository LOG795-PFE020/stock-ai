"""
Main script for stock price forecasting with Prophet.
"""

import argparse
import sys
import logging
import os
import json
import datetime as dt
from typing import Dict, Any, Optional

import data
import model
import output
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("forecast.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forecast stock prices using Prophet')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--force-reload', action='store_true', help='Force reload of data')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--quick', action='store_true', help='Use quicker tuning for faster results')
    parser.add_argument('--output-file', type=str, help='Output file for forecast (JSON)')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    parser.add_argument('--param-file', type=str, help='JSON file with model parameters to use')
    parser.add_argument('--model-file', type=str, help='Path to a saved model file to use')
    parser.add_argument('--use-saved-model', action='store_true', help='Use the latest saved model for this ticker')
    parser.add_argument('--no-save', action='store_true', help='Do not save the trained model')
    return parser.parse_args()

def forecast_stock(
    ticker: str, 
    force_reload: bool = False, 
    tune: bool = True, 
    quick_mode: bool = False,
    param_file: Optional[str] = None,
    model_file: Optional[str] = None,
    use_saved_model: bool = False,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline to forecast stock prices.
    
    Args:
        ticker: Stock ticker symbol
        force_reload: Whether to force reload data
        tune: Whether to tune model hyperparameters
        quick_mode: Whether to use quicker tuning for faster results
        param_file: Path to JSON file with model parameters to use
        model_file: Path to a saved model file to use
        use_saved_model: Whether to use the latest saved model for this ticker
        save_model: Whether to save the newly trained model
        
    Returns:
        Formatted forecast output
    """
    # Load model parameters from file if provided
    model_params = None
    if param_file:
        try:
            with open(param_file, 'r') as f:
                model_params = json.load(f)
            logger.info(f"Loaded model parameters from {param_file}: {model_params}")
            # When parameters are loaded from file, we don't need to tune
            tune = False
        except Exception as e:
            logger.error(f"Error loading parameters from {param_file}: {e}")
            return output.format_error_output(f"Error loading parameters: {str(e)}")
    
    # Validate ticker
    logger.info(f"Validating ticker: {ticker}")
    if not data.is_valid_ticker(ticker):
        logger.error(f"Invalid ticker: {ticker}")
        return output.format_error_output(f"Invalid ticker: {ticker}")
    logger.info(f"Ticker {ticker} is valid")
    
    # Get data
    try:
        stock_data = data.get_data_for_model(ticker, force_reload=force_reload)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return output.format_error_output(f"Error fetching data: {str(e)}")
    
    # Check if we need to load a saved model
    loaded_model = None
    
    # First priority: explicit model file
    if model_file:
        try:
            loaded_model = model.load_model(model_file)
            logger.info(f"Using provided model file: {model_file}")
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
            return output.format_error_output(f"Error loading model: {str(e)}")
    
    # Second priority: latest saved model if requested
    elif use_saved_model:
        latest_model_path = model.get_latest_model(ticker)
        if latest_model_path:
            try:
                loaded_model = model.load_model(latest_model_path)
                logger.info(f"Using latest saved model: {latest_model_path}")
            except Exception as e:
                logger.error(f"Error loading latest model for {ticker}: {e}")
                logger.warning("Will train a new model instead")
        else:
            logger.warning(f"No saved model found for {ticker}. Will train a new model.")
    
    # Generate forecasts if we have a loaded model
    if loaded_model:
        try:
            # Make a forecast using the loaded model
            forecast = model.generate_forecast(loaded_model, stock_data['full'])
            specific_forecasts = model.get_specific_forecasts(forecast)
            
            # Evaluate the loaded model on test data
            test_metrics = model.evaluate_model(loaded_model, stock_data['test'])
            
            # Get recent performance
            recent_performance = model.get_recent_performance(loaded_model, stock_data['full'], days=5)
            
            results = {
                'model': loaded_model,
                'forecast': forecast,
                'specific_forecasts': specific_forecasts,
                'test_metrics': test_metrics,
                'recent_performance': recent_performance,
                'model_path': model_file or model.get_latest_model(ticker),
                'params': {}  # We don't have the training parameters for a loaded model
            }
        except Exception as e:
            logger.error(f"Error using loaded model: {e}")
            logger.warning("Will train a new model instead")
            loaded_model = None
    
    # Train a new model if we need to
    if not loaded_model:
        try:
            results = model.train_and_forecast(
                ticker, 
                stock_data, 
                tune=tune, 
                model_params=model_params,
                quick_mode=quick_mode,
                save=save_model
            )
        except Exception as e:
            logger.error(f"Error in model training/forecasting: {e}")
            return output.format_error_output(f"Error in model training/forecasting: {str(e)}")
    
    # Format output
    forecasts = results['specific_forecasts']
    test_metrics = results['test_metrics']
    recent_performance = results.get('recent_performance', {})
    model_path = results.get('model_path', None)
    
    # Add model_path to the formatted output
    formatted_output = output.format_forecast_output(
        ticker, 
        forecasts, 
        test_metrics=test_metrics,
        recent_performance=recent_performance,
        model_path=model_path
    )
    
    # Save the best parameters to a file for future use
    if tune and results['params']:
        try:
            params_dir = os.path.join("data", "params")
            os.makedirs(params_dir, exist_ok=True)
            
            param_output_file = os.path.join(params_dir, f"{ticker.lower()}_params.json")
            with open(param_output_file, 'w') as f:
                json.dump(results['params'], f, indent=4)
            
            logger.info(f"Saved best parameters to {param_output_file}")
        except Exception as e:
            logger.warning(f"Could not save parameters to file: {e}")
    
    return formatted_output

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Make sure the data and outputs directories exist
    os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Run forecast
    logger.info(f"Forecasting stock prices for {args.ticker}")
    result = forecast_stock(
        ticker=args.ticker, 
        force_reload=args.force_reload, 
        tune=not args.no_tune,
        quick_mode=args.quick,
        param_file=args.param_file,
        model_file=args.model_file,
        use_saved_model=args.use_saved_model,
        save_model=not args.no_save
    )
    
    # Output results
    json_result = output.to_json_string(result)
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(json_result)
        logger.info(f"Forecast saved to {args.output_file}")
    
    # Generate a default output file if none specified
    elif 'error' not in result:
        default_output = f"outputs/{args.ticker.lower()}_forecast_{dt.datetime.now().strftime('%Y%m%d')}.json"
        with open(default_output, 'w') as f:
            f.write(json_result)
        logger.info(f"Forecast saved to {default_output}")
    
    # Print to console
    print(json_result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())