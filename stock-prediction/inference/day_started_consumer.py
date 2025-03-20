import pika
import json
import logging
import os
import requests
from typing import List
import threading
import time
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DayStartedConsumer:
    def __init__(self, host: str = 'localhost', port: int = 5672,
                 username: str = 'guest', password: str = 'guest',
                 api_host: str = 'localhost', api_port: str = '8000'):
        """Initialize the DayStarted event consumer"""
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(username, password),
            heartbeat=30,
            blocked_connection_timeout=10
        )
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.connection = None
        self.channel = None
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "specific")

    def connect(self) -> None:
        """Establish connection to RabbitMQ"""
        if self._is_shutting_down:
            return

        try:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

            # Declare the day-started exchange
            self.channel.exchange_declare(
                exchange='day-started-exchange',
                exchange_type='fanout',
                durable=True
            )

            # Create a temporary queue
            result = self.channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue

            # Bind to the exchange
            self.channel.queue_bind(
                exchange='day-started-exchange',
                queue=queue_name
            )

            # Set up consumer
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=self._handle_day_started,
                auto_ack=True
            )

            logger.info("Connected to RabbitMQ and ready to consume DayStarted events")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            if self.connection and not self.connection.is_closed:
                self.connection.close()

    def _handle_day_started(self, ch, method, properties, body):
        """Handle received DayStarted event"""
        try:
            # Parse the message
            message = json.loads(body)
            logger.info(f"Received message: {json.dumps(message, indent=2)}")

            # Extract timestamp from the DayStarted event
            timestamp = None
            if isinstance(message, dict):
                if 'message' in message:
                    timestamp = message['message'].get('Timestamp')
                elif 'Timestamp' in message:
                    timestamp = message['Timestamp']

            logger.info(f"Processing DayStarted event for timestamp: {timestamp}")

            # Get list of available symbols
            symbols = self._get_available_symbols()
            logger.info(f"Found {len(symbols)} symbols to process: {', '.join(symbols)}")
            
            predictions_made = 0
            for symbol in symbols:
                try:
                    # Request prediction from the API
                    logger.info(f"Requesting prediction for {symbol}...")
                    response = requests.get(
                        f"{self.api_base_url}/predict/next_day",
                        params={"symbol": symbol, "model_type": "lstm"}
                    )
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        logger.info(f"Successfully fetched prediction for {symbol}: {prediction['prediction']:.2f} (confidence: {prediction['confidence_score']:.2f})")
                        predictions_made += 1
                    else:
                        logger.error(f"Failed to get prediction for {symbol}: {response.text}")
                        
                    # Add small delay to prevent overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")

            logger.info(f"Completed processing DayStarted event. Made {predictions_made} predictions out of {len(symbols)} symbols.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message as JSON: {str(e)}")
            logger.error(f"Raw message: {body}")
        except Exception as e:
            logger.error(f"Error handling DayStarted event: {str(e)}")
            logger.exception("Full traceback:")

    def _get_available_symbols(self) -> List[str]:
        """Get list of symbols available for prediction based on available models"""
        try:
            # Get all model directories in the specific models directory
            symbols = []
            if os.path.exists(self.models_dir):
                # Look for model files in each symbol directory
                symbol_dirs = [d for d in os.listdir(self.models_dir) 
                             if os.path.isdir(os.path.join(self.models_dir, d))]
                
                for symbol in symbol_dirs:
                    model_path = os.path.join(self.models_dir, symbol, f"{symbol}_model.keras")
                    weights_path = os.path.join(self.models_dir, symbol, f"{symbol}_model.weights.h5")
                    scaler_path = os.path.join(self.models_dir, symbol, f"{symbol}_scaler.gz")
                    
                    # Only include symbols that have both model and scaler files
                    if ((os.path.exists(model_path) or os.path.exists(weights_path)) 
                            and os.path.exists(scaler_path)):
                        symbols.append(symbol)
            
            if not symbols:
                logger.warning("No models found in specific models directory")
                return ["AAPL", "GOOGL", "MSFT"]  # Fallback to test symbols
                
            logger.info(f"Found {len(symbols)} available symbols with models")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return ["AAPL", "GOOGL", "MSFT"]  # Fallback to test symbols

    def run(self):
        """Start consuming messages"""
        while not self._is_shutting_down:
            try:
                if not self.connection or self.connection.is_closed:
                    self.connect()
                
                logger.info("Starting to consume messages...")
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                self._is_shutting_down = True
                logger.info("Shutting down consumer...")
                if self.connection and not self.connection.is_closed:
                    self.connection.close()
                break
                
            except Exception as e:
                logger.error(f"Error in consumer: {str(e)}")
                if self.connection and not self.connection.is_closed:
                    self.connection.close()
                time.sleep(5)  # Wait before reconnecting

if __name__ == "__main__":
    # Get configuration from environment variables
    rabbitmq_host = os.environ.get("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port = int(os.environ.get("RABBITMQ_PORT", "5672"))
    rabbitmq_user = os.environ.get("RABBITMQ_USERNAME", "guest")
    rabbitmq_pass = os.environ.get("RABBITMQ_PASSWORD", "guest")
    api_host = os.environ.get("API_HOST", "localhost")
    api_port = os.environ.get("API_PORT", "8000")

    consumer = DayStartedConsumer(
        host=rabbitmq_host,
        port=rabbitmq_port,
        username=rabbitmq_user,
        password=rabbitmq_pass,
        api_host=api_host,
        api_port=api_port
    )
    
    consumer.run() 