import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .model_manager import ModelManager
from .rabbitmq_publisher import rabbitmq_publisher

logger = logging.getLogger(__name__)

class PredictionScheduler:
    def __init__(self, model_manager: ModelManager, symbols: List[str], 
                 interval_minutes: int = 60):
        """Initialize the prediction scheduler"""
        self.model_manager = model_manager
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_run: Dict[str, datetime] = {}
        
        # Initialize last run times
        for symbol in symbols:
            self._last_run[symbol] = datetime.min
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self._is_running:
            logger.warning("Scheduler is already running")
            return
            
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_scheduler,
            name="PredictionScheduler",
            daemon=True
        )
        self._thread.start()
        logger.info("Started prediction scheduler")
    
    def stop(self):
        """Stop the scheduler"""
        with self._lock:
            self._is_running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
                logger.info("Stopped prediction scheduler")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self._is_running:
            try:
                current_time = datetime.now()
                
                # Check each symbol
                for symbol in self.symbols:
                    last_run = self._last_run[symbol]
                    time_since_last_run = current_time - last_run
                    
                    # If enough time has passed, generate prediction
                    if time_since_last_run >= timedelta(minutes=self.interval_minutes):
                        try:
                            # Generate prediction
                            prediction = self.model_manager.predict(symbol)
                            
                            # Publish to RabbitMQ
                            publish_success = rabbitmq_publisher.publish_stock_quote(symbol, prediction)
                            
                            if publish_success:
                                logger.info(f"✅ Successfully published prediction for {symbol}")
                                prediction['rabbitmq_status'] = 'delivered'
                            else:
                                logger.warning(f"⚠️ Failed to confirm RabbitMQ delivery for {symbol}")
                                prediction['rabbitmq_status'] = 'unconfirmed'
                            
                            # Update last run time
                            self._last_run[symbol] = current_time
                            
                        except Exception as e:
                            logger.error(f"❌ Error generating prediction for {symbol}: {str(e)}")
                
                # Sleep for a short interval to prevent CPU overuse
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        with self._lock:
            return {
                'is_running': self._is_running,
                'symbols': self.symbols,
                'interval_minutes': self.interval_minutes,
                'last_runs': {
                    symbol: last_run.isoformat() 
                    for symbol, last_run in self._last_run.items()
                }
            }
    
    def update_symbols(self, new_symbols: List[str]):
        """Update the list of symbols to predict"""
        with self._lock:
            # Add new symbols
            for symbol in new_symbols:
                if symbol not in self._last_run:
                    self._last_run[symbol] = datetime.min
            
            # Remove old symbols
            for symbol in list(self._last_run.keys()):
                if symbol not in new_symbols:
                    del self._last_run[symbol]
            
            self.symbols = new_symbols
            logger.info(f"Updated symbols list: {new_symbols}")
    
    def update_interval(self, new_interval_minutes: int):
        """Update the prediction interval"""
        with self._lock:
            self.interval_minutes = new_interval_minutes
            logger.info(f"Updated prediction interval to {new_interval_minutes} minutes")
    
    def force_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Force a prediction for a specific symbol immediately"""
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in scheduler's symbol list")
            return None
            
        try:
            # Generate prediction
            prediction = self.model_manager.predict(symbol)
            
            # Publish to RabbitMQ
            publish_success = rabbitmq_publisher.publish_stock_quote(symbol, prediction)
            
            if publish_success:
                logger.info(f"✅ Successfully published forced prediction for {symbol}")
                prediction['rabbitmq_status'] = 'delivered'
            else:
                logger.warning(f"⚠️ Failed to confirm RabbitMQ delivery for {symbol}")
                prediction['rabbitmq_status'] = 'unconfirmed'
            
            # Update last run time
            self._last_run[symbol] = datetime.now()
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error generating forced prediction for {symbol}: {str(e)}")
            return None 