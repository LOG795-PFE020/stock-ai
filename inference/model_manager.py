import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model
import joblib
import json

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir: str = "models", processed_dir: str = "data/processed"):
        """Initialize the Model Manager"""
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.sequence_length = 60
        self.features = [
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Returns", "MA_5", "MA_20", "Volatility", "RSI", "MACD", "MACD_Signal"
        ]
        
        # Model storage
        self.general_model: Optional[Model] = None
        self.specific_models: Dict[str, Model] = {}
        self.specific_scalers: Dict[str, Any] = {}
        self.general_scalers: Dict[str, Any] = {}
        
        # Load models and scalers
        self.load_resources()
    
    def load_resources(self) -> None:
        """Load all models and scalers with proper error handling"""
        try:
            # Load specific models and scalers
            specific_dir = os.path.join(self.models_dir, "specific")
            if not os.path.exists(specific_dir):
                raise FileNotFoundError(f"Specific models directory {specific_dir} not found")
            
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
                    try:
                        if os.path.exists(model_keras_path):
                            model = tf.keras.models.load_model(model_keras_path)
                        elif os.path.exists(model_weights_path):
                            input_shape = (self.sequence_length, len(self.features))
                            model = self._build_specific_model(input_shape)
                            
                            # Load weights without optimizer state
                            try:
                                model.load_weights(model_weights_path, skip_mismatch=True)
                            except Exception as e:
                                logger.warning(f"Error loading weights for {symbol}: {str(e)}")
                                # Continue with untrained model
                        else:
                            skipped_count += 1
                            skipped_symbols.append(symbol)
                            continue
                    except Exception as e:
                        logger.error(f"Error loading model for {symbol}: {str(e)}")
                        error_count += 1
                        error_symbols.append(symbol)
                        continue
                    
                    # Load scaler and metadata
                    try:
                        scaler = joblib.load(scaler_path)
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading scaler/metadata for {symbol}: {str(e)}")
                        error_count += 1
                        error_symbols.append(symbol)
                        continue
                    
                    # Validate model architecture
                    if not self._validate_model_architecture(model):
                        logger.error(f"Invalid model architecture for {symbol}")
                        error_count += 1
                        error_symbols.append(symbol)
                        continue
                    
                    self.specific_models[symbol] = model
                    self.specific_scalers[symbol] = scaler
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
            
            if not self.specific_models:
                raise ValueError("No models were loaded")
                
        except Exception as e:
            logger.error(f"❌ Error loading resources: {str(e)}")
            raise
    
    def _validate_model_architecture(self, model: Model) -> bool:
        """Validate model architecture and weights"""
        try:
            # Check input shape
            input_shape = model.input_shape
            if not input_shape or input_shape[1:] != (self.sequence_length, len(self.features)):
                logger.error(f"Invalid input shape: {input_shape}")
                return False
            
            # Check output shape
            output_shape = model.output_shape
            if not output_shape or output_shape[1:] != (1,):
                logger.error(f"Invalid output shape: {output_shape}")
                return False
            
            # Check layer types and configurations
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    if layer.units < 8 or layer.units > 256:
                        logger.error(f"Invalid LSTM units: {layer.units}")
                        return False
                elif isinstance(layer, tf.keras.layers.Dense):
                    if layer.units < 1 or layer.units > 256:
                        logger.error(f"Invalid Dense units: {layer.units}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model architecture: {str(e)}")
            return False
    
    def _build_specific_model(self, input_shape: Tuple) -> Model:
        """Build a stock-specific model using functional API with enhanced regularization"""
        inputs = tf.keras.Input(shape=input_shape, name='sequence_input', dtype=tf.float32)
        
        # Add BatchNormalization at input with higher momentum
        x = tf.keras.layers.BatchNormalization(momentum=0.99, dtype=tf.float32)(inputs)
        
        # First LSTM layer with reduced complexity
        x = tf.keras.layers.LSTM(24, return_sequences=True,  # Using 24 units to match training code
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            dtype=tf.float32)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Reduced from 0.3
        
        # Second LSTM layer
        x = tf.keras.layers.LSTM(12,  # Reduced from 16
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                dtype=tf.float32)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Dense layers with ELU activation for better gradient flow
        x = tf.keras.layers.Dense(12, activation='elu',  # Changed from selu to elu
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                dtype=tf.float32)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.1)(x)  # Reduced from 0.2
        
        x = tf.keras.layers.Dense(6, activation='elu',  # Changed from selu to elu
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                dtype=tf.float32)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.1)(x)  # Reduced from 0.2
        
        outputs = tf.keras.layers.Dense(1, activation='linear', dtype=tf.float32)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='specific_stock_model')
        
        # Use a higher initial learning rate with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Increased from 0.0005
            clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Use MSE loss for better stability in early training
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Changed from Huber loss
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
        
        return model
    
    def get_latest_sequence(self, symbol: str) -> np.ndarray:
        """Retrieves and prepares the most recent sequence for a specific stock"""
        try:
            # Try to find the stock in the specific directory structure
            for sector in os.listdir(os.path.join(self.processed_dir, "specific")):
                sector_path = os.path.join(self.processed_dir, "specific", sector)
                if os.path.isdir(sector_path):
                    stock_file = os.path.join(sector_path, f"{symbol}_processed.csv")
                    if os.path.exists(stock_file):
                        df = pd.read_csv(stock_file)
                        break
            else:
                # If not found in specific directories, try the general processed directory
                stock_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
                if not os.path.exists(stock_file):
                    raise FileNotFoundError(f"No data found for symbol {symbol}")
                df = pd.read_csv(stock_file)
            
            # Get last SEQ_SIZE samples
            df = df.tail(self.sequence_length)
            
            # Check if all required features are present
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features for {symbol}: {missing_features}")
            
            # Prepare sequence with all required features
            sequence = df[self.features].values.reshape(1, self.sequence_length, len(self.features))
            return sequence
            
        except Exception as e:
            logger.error(f"Error getting latest sequence for {symbol}: {str(e)}")
            raise
    
    def get_original_price(self, symbol: str) -> float:
        """Get the last known original price for a symbol"""
        try:
            # First try raw data in Technology sector
            raw_file = os.path.join("data", "raw", "Technology", f"{symbol}_stock_price.csv")
            if os.path.exists(raw_file):
                df = pd.read_csv(raw_file)
                if not df.empty:
                    return float(df['Close'].iloc[-1])
            
            # Then try processed data
            processed_file = os.path.join(self.processed_dir, "specific", "Technology", f"{symbol}_processed.csv")
            if os.path.exists(processed_file):
                df = pd.read_csv(processed_file)
                if not df.empty:
                    # Get the last non-normalized close price
                    metadata_path = os.path.join(self.models_dir, "specific", symbol, f"{symbol}_scaler_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            close_idx = metadata['feature_names'].index('Close')
                            min_val = metadata['min_values'][close_idx]
                            max_val = metadata['max_values'][close_idx]
                            normalized_close = df['Close'].iloc[-1]
                            original_close = normalized_close * (max_val - min_val) + min_val
                            return float(original_close)
            
            # Finally try unified data
            unified_file = os.path.join("data", "raw", "unified", f"{symbol}_stock_price.csv")
            if os.path.exists(unified_file):
                df = pd.read_csv(unified_file)
                if not df.empty:
                    return float(df['Close'].iloc[-1])
                    
        except Exception as e:
            logger.warning(f"Could not find original price data: {str(e)}")
            logger.exception(e)  # Log the full traceback
        return None
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Generate prediction for a specific stock"""
        try:
            # Get the latest sequence
            sequence = self.get_latest_sequence(symbol)
            
            # Try to use specific model first
            if symbol in self.specific_models:
                model = self.specific_models[symbol]
                scaler = self.specific_scalers[symbol]
                model_type = "specific"
                
                # Load scaling metadata if available
                metadata_path = os.path.join(self.models_dir, "specific", symbol, f"{symbol}_scaler_metadata.json")
                scaling_metadata = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        scaling_metadata = json.load(f)
            else:
                raise ValueError(f"No model available for {symbol}")
            
            # Make prediction
            prediction = model.predict(sequence)
            
            # Get the last known values
            last_sequence = sequence[0, -1, :]  # Get the last row of the sequence
            close_index = self.features.index("Close")
            last_close = last_sequence[close_index]
            
            # Find the original close price from the raw data
            original_price = self.get_original_price(symbol)
            
            # Initialize variables for prediction details
            prediction_details = {
                'original_price': original_price,
                'raw_prediction': float(prediction[0, 0]),
                'scaling_method': 'metadata' if scaling_metadata else 'simple'
            }
            
            try:
                # Method 1: Use scaler with correct feature ordering
                if scaling_metadata:
                    # Use the exact feature order from training
                    features = scaling_metadata['feature_order']
                    scaler_ready = np.zeros((1, len(features)))
                    
                    # Fill in known values from the last sequence
                    for i, feature in enumerate(features):
                        if feature == "Close":
                            scaler_ready[0, i] = prediction[0, 0]
                        elif feature in self.features:
                            feat_idx = self.features.index(feature)
                            scaler_ready[0, i] = last_sequence[feat_idx]
                        else:
                            # For derived features, use their last known values
                            scaler_ready[0, i] = last_sequence[self.features.index(feature)] if feature in self.features else 0
                    
                    # Apply inverse transform
                    denormalized = scaler.inverse_transform(scaler_ready)
                    price = denormalized[0, features.index("Close")]
                else:
                    # Fallback to simpler scaling if metadata not available
                    price = self._simple_inverse_scale(prediction[0, 0], original_price, last_close)
                
                # Calculate relative change
                if original_price is not None:
                    relative_change = (price - original_price) / original_price
                    prediction_details['relative_change'] = float(relative_change)
                    prediction_details['change_percentage'] = float(relative_change * 100)
                    
                    # Apply more conservative price change limits
                    max_change = 0.05  # Maximum 5% change allowed
                    if abs(relative_change) > max_change:
                        logger.warning(f"Large price change detected for {symbol}: {relative_change*100:.2f}%")
                        # Calculate conservative estimate
                        conservative_price = original_price * (1 + np.sign(relative_change) * max_change)
                        prediction_details['status'] = 'large_change_detected'
                        prediction_details['original_prediction'] = float(price)
                        prediction_details['conservative_estimate'] = float(conservative_price)
                        prediction_details['max_change_allowed'] = max_change
                        # Use conservative estimate as final price
                        price = conservative_price
                    else:
                        prediction_details['status'] = 'within_normal_range'
                
                # Additional validation checks
                if price <= 0:
                    logger.warning(f"Invalid negative price detected for {symbol}: {price}")
                    price = original_price * 0.95  # Conservative fallback
                    prediction_details['status'] = 'invalid_price_detected'
                    prediction_details['fallback_price'] = float(price)
                
                # Check for extreme volatility
                if 'relative_change' in prediction_details:
                    volatility = self._calculate_volatility(sequence)
                    if volatility > 0.05:  # High volatility threshold
                        logger.warning(f"High volatility detected for {symbol}: {volatility:.2f}")
                        # Further reduce the prediction confidence
                        prediction_details['volatility'] = float(volatility)
                        prediction_details['status'] = 'high_volatility'
                
            except Exception as e:
                logger.warning(f"Error in scaling inverse transform for {symbol}: {str(e)}")
                price = self._simple_inverse_scale(prediction[0, 0], original_price, last_close)
                prediction_details['status'] = 'fallback_to_simple'
                prediction_details['error'] = str(e)
            
            # Calculate confidence score with enhanced validation
            confidence_score = self._calculate_confidence_score(sequence, prediction)
            
            # Adjust confidence score based on various factors
            if 'relative_change' in prediction_details:
                change_factor = abs(prediction_details['relative_change'])
                if change_factor > 0.05:  # More than 5% change
                    confidence_score *= (1 - (change_factor - 0.05))  # Reduce confidence for large changes
            
            if prediction_details.get('status') == 'high_volatility':
                confidence_score *= 0.8  # Reduce confidence for high volatility
            
            if prediction_details.get('status') == 'invalid_price_detected':
                confidence_score *= 0.7  # Further reduce confidence for invalid prices
            
            # Ensure confidence score is within bounds
            confidence_score = max(0.1, min(0.95, confidence_score))
            
            result = {
                'prediction': float(price),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': "1.0.0",  # TODO: Implement proper versioning
                'model_type': f'lstm_{model_type}',
                'prediction_details': prediction_details
            }
            
            # Log a single consolidated message for the prediction
            status = prediction_details.get('status', 'normal')
            warnings = []
            if status == 'large_change_detected':
                warnings.append(f"large change ({prediction_details['change_percentage']:.2f}%)")
            if status == 'high_volatility':
                warnings.append(f"high volatility ({prediction_details['volatility']:.2f})")
            if status == 'invalid_price_detected':
                warnings.append("invalid price")
            
            if warnings:
                logger.warning(f"⚠️ {symbol}: {' | '.join(warnings)}")
            else:
                logger.info(f"✅ {symbol}: Predicted {price:.2f} (confidence: {confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error predicting {symbol}: {str(e)}")
            raise
    
    def _simple_inverse_scale(self, predicted_normalized: float, original_price: float, last_close_normalized: float) -> float:
        """Simple scaling based on relative change with conservative limits"""
        try:
            if original_price is None:
                return predicted_normalized  # Return as is if no reference point
                
            # Calculate the relative change from the normalized values
            relative_change = (predicted_normalized - last_close_normalized) / last_close_normalized
            
            # Apply more conservative price change limits
            max_change = 0.05  # Maximum 5% change allowed
            relative_change = np.clip(relative_change, -max_change, max_change)
            
            # Apply the clipped relative change to the original price
            result = original_price * (1 + relative_change)
            
            logger.info(f"Simple scaling: orig={original_price}, pred_norm={predicted_normalized}, last_norm={last_close_normalized}, change={relative_change}, result={result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in simple inverse scaling: {str(e)}")
            return original_price * 0.95  # Conservative fallback
    
    def _calculate_confidence_score(self, sequence: np.ndarray, prediction: np.ndarray) -> float:
        """Calculate confidence score for the prediction"""
        # TODO: Implement more sophisticated confidence calculation
        return 0.85
    
    def _calculate_volatility(self, sequence: np.ndarray) -> float:
        """Calculate volatility from the sequence"""
        try:
            # Get the Close prices from the sequence
            close_index = self.features.index("Close")
            close_prices = sequence[0, :, close_index]
            
            # Calculate returns
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def batch_predict(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate predictions for multiple stocks"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.predict(symbol)
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
                results[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        return results 