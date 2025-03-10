from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model, Model
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Dict, List, Any, Tuple
import logging
import os
import tensorflow as tf
import sys
import platform
from prophet import Prophet
from .rabbitmq_publisher import rabbitmq_publisher
from .model_manager import ModelManager
from .scheduler import PredictionScheduler
import json
from keras.layers import LSTM, Dropout, Dense, Input, BatchNormalization

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce TensorFlow logging verbosity
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors

# Version information
PYTHON_VERSION = platform.python_version()
TF_VERSION = tf.__version__
KERAS_VERSION = tf.keras.__version__

logger.info(f"Starting Stock Price Prediction API Server")
logger.info(f"Python {PYTHON_VERSION} | TensorFlow {TF_VERSION} | Keras {KERAS_VERSION}")

app = Flask(__name__)
api = Api(
    app, 
    title='Stock Price Prediction API',
    version='1.0',
    description='A production-ready API for predicting stock prices using LSTM',
    doc='/docs',
    default='Stock Prediction API',
    default_label='ML-powered stock price prediction endpoints'
)

# Create namespaces for better organization
ns_home = api.namespace('', description='Home page')
ns_predict = api.namespace('predict', description='Stock price prediction operations')
ns_health = api.namespace('health', description='Health and monitoring operations')
ns_meta = api.namespace('meta', description='API metadata and information')
ns_scheduler = api.namespace('scheduler', description='Prediction scheduler operations')

# Response Models
error_model = api.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'status_code': fields.Integer(required=True, description='HTTP status code'),
    'timestamp': fields.DateTime(required=True, description='Error timestamp')
})

prediction_model = api.model('Prediction', {
    'prediction': fields.Float(required=True, description='Predicted stock price in USD'),
    'timestamp': fields.DateTime(required=True, description='Prediction timestamp'),
    'confidence_score': fields.Float(description='Model confidence score (0-1)', min=0, max=1),
    'model_version': fields.String(description='Version of the ML model used'),
    'model_type': fields.String(description='Type of model used (general or specific)')
})

predictions_model = api.model('Predictions', {
    'predictions': fields.List(
        fields.Nested(prediction_model), 
        description='List of predictions'
    ),
    'start_date': fields.DateTime(description='Start date of predictions'),
    'end_date': fields.DateTime(description='End date of predictions'),
    'average_confidence': fields.Float(description='Average confidence score')
})

meta_model = api.model('MetaInfo', {
    'model_info': fields.Nested(api.model('ModelInfo', {
        'version': fields.String(description='Model version'),
        'last_trained': fields.DateTime(description='Last training timestamp'),
        'accuracy_score': fields.Float(description='Model accuracy score')
    })),
    'api_info': fields.Nested(api.model('ApiInfo', {
        'version': fields.String(description='API version'),
        'uptime': fields.String(description='API uptime'),
        'requests_served': fields.Integer(description='Total requests served')
    }))
})

scheduler_status_model = api.model('SchedulerStatus', {
    'is_running': fields.Boolean(description='Whether the scheduler is running'),
    'symbols': fields.List(fields.String, description='List of symbols being predicted'),
    'interval_minutes': fields.Integer(description='Prediction interval in minutes'),
    'last_runs': fields.Raw(description='Last prediction times for each symbol')
})

# Global variables
GENERAL_MODEL = None
SPECIFIC_MODELS = {}
SPECIFIC_SCALERS = {}
GENERAL_SCALERS = {}
PROPHET_MODELS = {}  # Store Prophet models for each symbol
SEQ_SIZE = 60
FEATURES = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Returns", "MA_5", "MA_20", "Volatility", "RSI", "MACD", "MACD_Signal"
]
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()
REQUEST_COUNT = 0

# Initialize ModelManager and PredictionScheduler
model_manager = ModelManager()
scheduler = PredictionScheduler(
    model_manager=model_manager,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CSCO', 'INTC'],
    interval_minutes=60
)

class ModelNotLoadedError(Exception):
    """Raised when the model is not properly loaded"""
    pass

def load_prophet_model(symbol: str) -> Prophet:
    """Load or create a Prophet model for a given symbol"""
    try:
        # Check if model already exists in memory
        if symbol in PROPHET_MODELS:
            return PROPHET_MODELS[symbol]
            
        # Try to load from disk
        prophet_model_path = os.path.join("models", "prophet", f"{symbol}_prophet.json")
        if os.path.exists(prophet_model_path):
            with open(prophet_model_path, 'r') as fin:
                model_data = json.load(fin)
            
            # Create model with saved parameters
            model = Prophet(**model_data['params'])
            
            # Add regressors
            for regressor in model_data['regressors']:
                model.add_regressor(regressor)
            
            # Convert last data back to DataFrame
            df = pd.DataFrame(model_data['last_data'])
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Fit the model on the last data
            model.fit(df)
            
            PROPHET_MODELS[symbol] = model
            logger.info(f"✅ Loaded Prophet model for {symbol}")
            return model
            
        # Create new model if not found
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        PROPHET_MODELS[symbol] = model
        logger.info(f"✅ Created new Prophet model for {symbol}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading Prophet model for {symbol}: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load Prophet model for {symbol}") from e

def load_resources() -> None:
    """Load ML models and scalers with proper error handling"""
    global GENERAL_MODEL, SPECIFIC_MODELS, SPECIFIC_SCALERS, GENERAL_SCALERS
    
    try:
        # Load specific models and scalers
        specific_dir = "models/specific"
        if not os.path.exists(specific_dir):
            raise ModelNotLoadedError(f"Specific models directory {specific_dir} not found")
            
        loaded_count = 0
        skipped_count = 0
        error_count = 0
        loaded_symbols = []
        skipped_symbols = []
        error_symbols = []
        
        # Get all symbol directories
        symbol_dirs = [
            d for d in os.listdir(specific_dir) 
            if os.path.isdir(os.path.join(specific_dir, d))
        ]
        
        logger.info(f"Loading models for {len(symbol_dirs)} symbols...")
        
        for symbol in symbol_dirs:
            try:
                symbol_dir = os.path.join(specific_dir, symbol)
                model_keras_path = os.path.join(symbol_dir, f"{symbol}_model.keras")
                model_weights_path = os.path.join(symbol_dir, f"{symbol}_model.weights.h5")
                scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
                metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
                
                # Skip if we don't have both scaler and metadata
                if not (os.path.exists(scaler_path) and os.path.exists(metadata_path)):
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Try to load the model
                if os.path.exists(model_keras_path):
                    model = load_model(model_keras_path)
                elif os.path.exists(model_weights_path):
                    # Build the model with the correct input shape
                    input_shape = (SEQ_SIZE, len(FEATURES))
                    model = build_specific_model(input_shape)
                    # Load the weights without optimizer state
                    model.load_weights(model_weights_path, skip_mismatch=True)
                else:
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Load scaler and metadata
                scaler = joblib.load(scaler_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                SPECIFIC_MODELS[symbol] = model
                SPECIFIC_SCALERS[symbol] = scaler
                loaded_count += 1
                loaded_symbols.append(symbol)
                
            except Exception as e:
                error_count += 1
                error_symbols.append(symbol)
                logger.error(f"❌ Error loading model for {symbol}: {str(e)}")
                continue
        
        # Print summary
        logger.info(f"\n=== Model Loading Summary ===")
        logger.info(f"✅ Successfully loaded {loaded_count} models")
        if skipped_count > 0:
            logger.info(f"⚠️ Skipped {skipped_count} models: {', '.join(skipped_symbols)}")
        if error_count > 0:
            logger.info(f"❌ Failed to load {error_count} models: {', '.join(error_symbols)}")
        logger.info("===========================\n")
        
        if not SPECIFIC_MODELS:
            raise ModelNotLoadedError("No models were loaded")
            
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        raise ModelNotLoadedError("Failed to load model resources") from e

def build_specific_model(input_shape: Tuple) -> Model:
    """Build a stock-specific model using functional API"""
    inputs = Input(shape=input_shape, name='sequence_input', dtype=tf.float32)
    
    # First LSTM layer with reduced complexity
    x = LSTM(24, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            dtype=tf.float32)(inputs)
    x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
    x = Dropout(0.2)(x)
    
    # Second LSTM layer
    x = LSTM(12,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            dtype=tf.float32)(x)
    x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
    x = Dropout(0.2)(x)
    
    # Dense layers with ELU activation
    x = Dense(12, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            kernel_initializer='glorot_uniform',
            dtype=tf.float32)(x)
    x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
    x = Dropout(0.1)(x)
    
    x = Dense(6, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            kernel_initializer='glorot_uniform',
            dtype=tf.float32)(x)
    x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(1, activation='linear', dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='specific_stock_model')
    
    # Use a higher initial learning rate with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=0.5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def get_latest_sequence(symbol: str) -> np.ndarray:
    """Retrieves and prepares the most recent sequence for a specific stock"""
    try:
        # Try to find the stock in the specific directory structure
        for sector in os.listdir(os.path.join("data", "processed", "specific")):
            sector_path = os.path.join("data", "processed", "specific", sector)
            if os.path.isdir(sector_path):
                stock_file = os.path.join(sector_path, f"{symbol}_processed.csv")
                if os.path.exists(stock_file):
                    df = pd.read_csv(stock_file)
                    break
        else:
            # If not found in specific directories, try the general processed directory
            stock_file = os.path.join("data", "processed", f"{symbol}_processed.csv")
            if not os.path.exists(stock_file):
                raise FileNotFoundError(f"No data found for symbol {symbol}")
            df = pd.read_csv(stock_file)
        
        # Get last SEQ_SIZE samples
        df = df.tail(SEQ_SIZE)
        
        # Check if all required features are present
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for {symbol}: {missing_features}")
        
        # Prepare sequence with all required features
        sequence = df[FEATURES].values.reshape(1, SEQ_SIZE, len(FEATURES))
        logger.info(f"Successfully prepared sequence for {symbol} with shape {sequence.shape}")
        return sequence
        
    except Exception as e:
        logger.error(f"Error getting latest sequence for {symbol}: {str(e)}")
        raise

@ns_home.route('/')
class Home(Resource):
    @api.doc(
        responses={
            200: 'Welcome message',
        }
    )
    def get(self):
        """Get welcome message and basic API information"""
        return {
            'message': 'Welcome to the Stock Price Prediction API',
            'version': '1.0',
            'documentation': '/docs',
            'endpoints': {
                'health_check': '/health',
                'next_day_prediction': '/predict/next_day',
                'next_week_prediction': '/predict/next_week',
                'scheduler_status': '/scheduler/status'
            }
        }

@ns_meta.route('/info')
class MetaInfo(Resource):
    @api.doc(
        description='Get API and model metadata',
        responses={
            HTTPStatus.OK: ('Success', meta_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    @api.marshal_with(meta_model)
    def get(self) -> Dict[str, Any]:
        """Get API and model metadata including version, uptime, and statistics"""
        global REQUEST_COUNT
        try:
            return {
                'model_info': {
                    'version': MODEL_VERSION,
                    'last_trained': datetime(2024, 3, 1),  # Example date
                    'accuracy_score': 0.95  # Example score
                },
                'api_info': {
                    'version': '1.0',
                    'uptime': str(datetime.now() - START_TIME),
                    'requests_served': REQUEST_COUNT
                }
            }
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_health.route('/health')
class HealthCheck(Resource):
    @api.doc(
        description='Check API health status',
        responses={
            HTTPStatus.OK: 'API is healthy',
            HTTPStatus.SERVICE_UNAVAILABLE: 'API is unhealthy'
        }
    )
    def get(self) -> Dict[str, Any]:
        """Comprehensive health check of the API and its dependencies"""
        try:
            # Get scheduler status
            scheduler_status = scheduler.get_status()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'model_manager': 'healthy',
                    'scheduler': 'running' if scheduler_status['is_running'] else 'stopped',
                    'rabbitmq': 'healthy'
                },
                'scheduler_status': scheduler_status
            }, HTTPStatus.OK
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, HTTPStatus.SERVICE_UNAVAILABLE

@ns_predict.route('/next_day')
class NextDayPrediction(Resource):
    @api.doc(
        description='Get stock price prediction for the next trading day',
        params={
            'symbol': {
                'in': 'query',
                'description': 'Stock symbol (e.g., AAPL, GOOGL, MSFT)',
                'type': 'string',
                'required': True
            },
            'model_type': {
                'in': 'query',
                'description': 'Model type to use for prediction (lstm or prophet)',
                'type': 'string',
                'enum': ['lstm', 'prophet'],
                'default': 'lstm'
            }
        },
        responses={
            HTTPStatus.OK: ('Successful prediction', prediction_model),
            HTTPStatus.BAD_REQUEST: ('Invalid request', error_model),
            HTTPStatus.NOT_FOUND: ('Model not found', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Prediction error', error_model)
        }
    )
    @api.marshal_with(prediction_model)
    def get(self) -> Dict[str, Any]:
        """Get detailed stock price prediction for the next trading day"""
        global REQUEST_COUNT
        
        # Get stock symbol and model type from query parameters
        stock_symbol = request.args.get('symbol') or request.headers.get('X-Fields')
        model_type = request.args.get('model_type', 'lstm').lower()
        
        if not stock_symbol:
            api.abort(
                HTTPStatus.BAD_REQUEST,
                "Stock symbol is required. Provide it as a query parameter 'symbol' or header 'X-Fields'"
            )
            
        if model_type not in ['lstm', 'prophet']:
            api.abort(
                HTTPStatus.BAD_REQUEST,
                "Invalid model_type. Must be either 'lstm' or 'prophet'"
            )
        
        try:
            REQUEST_COUNT += 1
            logger.info(f"Processing {model_type} prediction request for {stock_symbol}")
            
            if model_type == 'prophet':
                return self._get_prophet_prediction(stock_symbol)
            else:
                return model_manager.predict(stock_symbol)
                
        except Exception as e:
            logger.error(f"Prediction error for {stock_symbol}: {str(e)}")
            api.abort(
                HTTPStatus.INTERNAL_SERVER_ERROR, 
                f"Prediction error: {str(e)}"
            )
    
    def _get_prophet_prediction(self, stock_symbol: str) -> Dict[str, Any]:
        """Get prediction using Prophet model"""
        try:
            # Get historical data for the stock
            stock_file = None
            
            # First check in Technology sector
            tech_file = os.path.join("data", "raw", "Technology", f"{stock_symbol}_stock_price.csv")
            if os.path.exists(tech_file):
                stock_file = tech_file
            
            # Then check in processed/specific directories
            if stock_file is None:
                for sector in os.listdir(os.path.join("data", "processed", "specific")):
                    sector_path = os.path.join("data", "processed", "specific", sector)
                    if os.path.isdir(sector_path):
                        temp_file = os.path.join(sector_path, f"{stock_symbol}_processed.csv")
                        if os.path.exists(temp_file):
                            stock_file = temp_file
                            break
            
            # Finally check in raw data root
            if stock_file is None:
                raw_file = os.path.join("data", "raw", f"{stock_symbol}_stock_price.csv")
                if os.path.exists(raw_file):
                    stock_file = raw_file
            
            if stock_file is None:
                raise FileNotFoundError(f"No data found for symbol {stock_symbol}")
            
            # Load and prepare data
            df = pd.read_csv(stock_file)
            df['ds'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
            df['y'] = df['Close']
            
            # Load or create cached model
            model = load_prophet_model(stock_symbol)
            
            # Check if we need to update the model with new data
            last_fit_date = pd.to_datetime(model.history['ds'].max())
            latest_data_date = df['ds'].max()
            
            if latest_data_date > last_fit_date:
                # Only fit on new data
                new_data = df[df['ds'] > last_fit_date]
                model.fit(new_data)
                
                # Save updated model
                prophet_model_path = os.path.join("models", "prophet", f"{stock_symbol}_prophet.json")
                os.makedirs(os.path.dirname(prophet_model_path), exist_ok=True)
                
                model_data = {
                    'params': {
                        'changepoint_prior_scale': model.changepoint_prior_scale,
                        'seasonality_prior_scale': model.seasonality_prior_scale,
                        'seasonality_mode': model.seasonality_mode,
                        'daily_seasonality': model.daily_seasonality,
                        'weekly_seasonality': model.weekly_seasonality,
                        'yearly_seasonality': model.yearly_seasonality
                    },
                    'regressors': model.extra_regressors.keys(),
                    'last_data': model.history.to_dict('records')
                }
                
                with open(prophet_model_path, 'w') as fout:
                    json.dump(model_data, fout)
            
            # Make prediction
            future = model.make_future_dataframe(periods=1)
            
            # Add regressors to future dataframe
            if 'volume' in model.extra_regressors:
                future['volume'] = df['Volume'].iloc[-1]
                
            if 'rsi' in model.extra_regressors:
                future['rsi'] = df['RSI'].iloc[-1]
            
            forecast = model.predict(future)
            
            # Get the prediction for tomorrow
            prediction = forecast.iloc[-1]['yhat']
            
            # Calculate confidence score based on uncertainty intervals
            lower = forecast.iloc[-1]['yhat_lower']
            upper = forecast.iloc[-1]['yhat_upper']
            interval_width = upper - lower
            last_price = df['Close'].iloc[-1]
            confidence_score = max(0, min(1, 1 - (interval_width / (4 * last_price))))
            
            result = {
                'prediction': float(prediction),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': MODEL_VERSION,
                'model_type': 'prophet'
            }
            
            # Publish to RabbitMQ
            try:
                publish_success = rabbitmq_publisher.publish_stock_quote(stock_symbol, result)
                if publish_success:
                    logger.info(f"✅ {stock_symbol}: Prophet prediction {prediction:.2f} (confidence: {confidence_score:.2f})")
                else:
                    logger.warning(f"⚠️ {stock_symbol}: Failed to publish prediction")
            except Exception as e:
                logger.error(f"❌ {stock_symbol}: Failed to publish prediction - {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prophet prediction error for {stock_symbol}: {str(e)}")
            raise

@ns_predict.route('/batch')
class BatchPrediction(Resource):
    @api.doc(
        description='Get predictions for multiple stocks',
        params={
            'symbols': {
                'in': 'query',
                'description': 'Comma-separated list of stock symbols',
                'type': 'string',
                'required': True
            }
        },
        responses={
            HTTPStatus.OK: ('Successful predictions', predictions_model),
            HTTPStatus.BAD_REQUEST: ('Invalid request', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Prediction error', error_model)
        }
    )
    @api.marshal_with(predictions_model)
    def get(self) -> Dict[str, Any]:
        """Get predictions for multiple stocks"""
        global REQUEST_COUNT
        
        # Get symbols from query parameters
        symbols_str = request.args.get('symbols')
        if not symbols_str:
            api.abort(
                HTTPStatus.BAD_REQUEST,
                "Stock symbols are required. Provide them as a comma-separated list in the 'symbols' query parameter"
            )
        
        symbols = [s.strip() for s in symbols_str.split(',')]
        
        try:
            REQUEST_COUNT += 1
            logger.info(f"Processing batch prediction request for {len(symbols)} symbols")
            
            # Get predictions using ModelManager
            results = model_manager.batch_predict(symbols)
            
            # Format response
            predictions = []
            total_confidence = 0
            valid_predictions = 0
            
            for symbol, result in results.items():
                if 'error' not in result:
                    predictions.append({
                        'symbol': symbol,
                        'prediction': result['prediction'],
                        'timestamp': result['timestamp'],
                        'confidence_score': result['confidence_score'],
                        'model_version': result['model_version'],
                        'model_type': result['model_type']
                    })
                    total_confidence += result['confidence_score']
                    valid_predictions += 1
            
            return {
                'predictions': predictions,
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(days=1),
                'average_confidence': total_confidence / valid_predictions if valid_predictions > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            api.abort(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Batch prediction error: {str(e)}"
            )

@ns_scheduler.route('/status')
class SchedulerStatus(Resource):
    @api.doc(
        description='Get current scheduler status',
        responses={
            HTTPStatus.OK: ('Success', scheduler_status_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    @api.marshal_with(scheduler_status_model)
    def get(self) -> Dict[str, Any]:
        """Get current status of the prediction scheduler"""
        try:
            return scheduler.get_status()
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_scheduler.route('/start')
class StartScheduler(Resource):
    @api.doc(
        description='Start the prediction scheduler',
        responses={
            HTTPStatus.OK: 'Scheduler started successfully',
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    def post(self):
        """Start the prediction scheduler"""
        try:
            scheduler.start()
            return {'message': 'Scheduler started successfully'}, HTTPStatus.OK
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_scheduler.route('/stop')
class StopScheduler(Resource):
    @api.doc(
        description='Stop the prediction scheduler',
        responses={
            HTTPStatus.OK: 'Scheduler stopped successfully',
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    def post(self):
        """Stop the prediction scheduler"""
        try:
            scheduler.stop()
            return {'message': 'Scheduler stopped successfully'}, HTTPStatus.OK
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_scheduler.route('/symbols')
class UpdateSymbols(Resource):
    @api.doc(
        description='Update the list of symbols to predict',
        params={
            'symbols': {
                'in': 'query',
                'description': 'Comma-separated list of stock symbols',
                'type': 'string',
                'required': True
            }
        },
        responses={
            HTTPStatus.OK: 'Symbols updated successfully',
            HTTPStatus.BAD_REQUEST: ('Invalid request', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    def put(self):
        """Update the list of symbols to predict"""
        try:
            symbols_str = request.args.get('symbols')
            if not symbols_str:
                api.abort(
                    HTTPStatus.BAD_REQUEST,
                    "Stock symbols are required. Provide them as a comma-separated list in the 'symbols' query parameter"
                )
            
            symbols = [s.strip() for s in symbols_str.split(',')]
            scheduler.update_symbols(symbols)
            return {'message': 'Symbols updated successfully'}, HTTPStatus.OK
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_scheduler.route('/interval')
class UpdateInterval(Resource):
    @api.doc(
        description='Update the prediction interval',
        params={
            'minutes': {
                'in': 'query',
                'description': 'New interval in minutes',
                'type': 'integer',
                'required': True
            }
        },
        responses={
            HTTPStatus.OK: 'Interval updated successfully',
            HTTPStatus.BAD_REQUEST: ('Invalid request', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    def put(self):
        """Update the prediction interval"""
        try:
            minutes = request.args.get('minutes')
            if not minutes or not minutes.isdigit():
                api.abort(
                    HTTPStatus.BAD_REQUEST,
                    "Valid interval in minutes is required"
                )
            
            scheduler.update_interval(int(minutes))
            return {'message': 'Interval updated successfully'}, HTTPStatus.OK
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_scheduler.route('/force/<string:symbol>')
class ForcePrediction(Resource):
    @api.doc(
        description='Force a prediction for a specific symbol',
        responses={
            HTTPStatus.OK: ('Success', prediction_model),
            HTTPStatus.NOT_FOUND: ('Symbol not found', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    @api.marshal_with(prediction_model)
    def post(self, symbol: str):
        """Force a prediction for a specific symbol"""
        try:
            result = scheduler.force_prediction(symbol)
            if result is None:
                api.abort(HTTPStatus.NOT_FOUND, f"Symbol {symbol} not found in scheduler")
            return result
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

if __name__ == '__main__':
    load_resources()
    try:
        # Start the scheduler
        scheduler.start()
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=8000, debug=False)
    finally:
        # Ensure RabbitMQ connection is closed when the server stops
        rabbitmq_publisher.close()