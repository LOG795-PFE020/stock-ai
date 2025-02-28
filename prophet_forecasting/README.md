# Stock Price Prediction with Prophet

This project implements a stock price prediction model using the Prophet library. It forecasts closing stock prices at three specific time horizons: 24 hours, 48 hours, and one week.

## Features

- Retrieves historical stock price data from Yahoo Finance
- Implements data caching to reduce API calls
- Uses Prophet for time series forecasting
- Supports hyperparameter tuning to optimize model performance
- Provides closing price predictions for 24h, 48h, and 1-week horizons
- Outputs predictions in JSON format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/prophet_forecasting.git
cd prophet_forecasting
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To generate a forecast for a specific stock ticker:

```bash
python main.py AAPL
```

This will output a JSON object with forecasts for the closing price at 24 hours, 48 hours, and one week:

```json
{
    "ticker": "AAPL",
    "forecasts": {
        "24h": 185.50,
        "48h": 186.20,
        "1w": 188.00
    },
    "date": "2023-10-25"
}
```

### Command Line Options

- `--force-reload`: Force reload of data instead of using cached data
- `--no-tune`: Skip hyperparameter tuning (faster but potentially less accurate)
- `--output-file FILE`: Save the forecast to a file instead of printing to console
- `--quiet`: Suppress console output except for the final JSON result

Example:
```bash
python main.py AAPL --force-reload --output-file forecast.json
```

## Project Structure

- `main.py`: Main entry point for the application
- `data.py`: Functions for retrieving and processing stock data
- `model.py`: Functions for training, evaluating, and generating forecasts
- `output.py`: Functions for formatting the output
- `config.py`: Configuration parameters
- `tests/`: Unit tests

## Testing

Run tests with pytest:

```bash
pytest
```

## Implementation Details

### Data Retrieval

- Uses the yfinance library to fetch historical stock data from Yahoo Finance
- Retrieves 5 years of historical data by default
- Implements caching to reduce API calls

### Data Preparation

- Splits the historical data into training (80%), validation (10%), and test (10%) sets
- Prepares the data for Prophet's required format (ds/y columns)

### Model Training

- Trains an initial model on the training set
- Tunes hyperparameters using the validation set
- Retrains the final model on the combined training and validation sets
- Evaluates the model on the test set

### Forecasting

- Generates a 7-day forecast
- Extracts the predictions for 24h, 48h, and 1-week horizons
- Formats the predictions into a standardized JSON structure

## License

MIT License