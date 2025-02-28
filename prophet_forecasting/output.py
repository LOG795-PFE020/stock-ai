"""
Module for formatting model outputs.
"""

import json
import datetime as dt
import os
from typing import Dict, Any, Optional

def format_forecast_output(
    ticker: str,
    forecasts: Dict[str, float],
    date: dt.date = None,
    test_metrics: Optional[Dict[str, float]] = None,
    recent_performance: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format forecast data into the required output structure.
    
    Args:
        ticker: Stock ticker symbol
        forecasts: Dictionary with forecast values for 24h, 48h, and 1w
        date: Date of the forecast (defaults to today)
        test_metrics: Optional evaluation metrics on test data
        recent_performance: Optional details on recent performance
        
    Returns:
        Structured forecast output
    """
    # Use today's date if not provided
    if date is None:
        date = dt.datetime.now().date()
    
    # Create the basic output structure
    output = {
        "ticker": ticker.upper(),
        "forecasts": {
            "24h": forecasts["24h"],
            "48h": forecasts["48h"],
            "1w": forecasts["1w"]
        },
        "date": date.strftime("%Y-%m-%d")
    }
    
    # Add model performance metrics if available
    if test_metrics:
        # Round all metric values to 4 decimal places
        rounded_metrics = {k: round(float(v), 4) if isinstance(v, (int, float)) else v 
                           for k, v in test_metrics.items()}
        output["model_performance"] = rounded_metrics
    
    # Add recent comparisons if available
    if recent_performance and "comparison" in recent_performance:
        output["recent_predictions"] = {
            "days_analyzed": recent_performance["days"],
            "within_interval_percentage": round(recent_performance["within_interval_percentage"], 2),
            "comparisons": recent_performance["comparison"],
            "metrics": {k: round(float(v), 4) if isinstance(v, (int, float)) else v 
                       for k, v in recent_performance["metrics"].items()}
        }
    
    # Add model path if available
    if model_path:
        # Only store the filename, not the full path for security reasons
        output["model_file"] = os.path.basename(model_path)
    
    return output

def format_error_output(error_message: str) -> Dict[str, str]:
    """
    Format an error message into a JSON object.
    
    Args:
        error_message: The error message
        
    Returns:
        Error message in JSON format
    """
    return {"error": error_message}

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.
    """
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def to_json_string(data: Dict) -> str:
    """
    Convert a dictionary to a JSON string.
    
    Args:
        data: Dictionary to convert
        
    Returns:
        JSON string
    """
    return json.dumps(data, indent=4, cls=NumpyEncoder)