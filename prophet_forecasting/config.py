"""
Configuration parameters for the stock prediction model.
"""

# Data parameters
DATA_CACHE_DIR = "data/cache"
DATA_MODELS_DIR = "data/models"
DATA_PARAMS_DIR = "data/params"
DATA_YEARS = 5  # Number of years of historical data to retrieve

# Model parameters
DEFAULT_CHANGEPOINT_PRIOR_SCALE = 0.01
DEFAULT_SEASONALITY_MODE = "multiplicative"
DEFAULT_DAILY_SEASONALITY = True
DEFAULT_WEEKLY_SEASONALITY = True
DEFAULT_YEARLY_SEASONALITY = True
DEFAULT_SEASONALITY_PRIOR_SCALE = 10.0
DEFAULT_HOLIDAYS_PRIOR_SCALE = 10.0
DEFAULT_CHANGEPOINT_RANGE = 0.9

# Train/validation/test split ratios
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1  # This is redundant but kept for clarity

# Forecast parameters
FORECAST_DAYS = 7  # Number of days to forecast