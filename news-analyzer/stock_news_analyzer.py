#!/usr/bin/env python
# coding: utf-8

"""
Stock News Analyzer API - Scrapes news articles for a stock ticker,
performs sentiment analysis using FinBERT-tone, and publishes to RabbitMQ.
"""

import json
import random
import os
import sys
from datetime import datetime
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from newspaper import Article, Config
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import nltk
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import requests

sys.path.append(os.path.dirname(__file__))
from news_publisher import NewsPublisher

# Logging setup
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'news_analyzer.log'))
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        logger.info(f"Downloading NLTK {resource}...")
        nltk.download(resource, quiet=True)

# Flask and API setup
app = Flask(__name__)
api = Api(app, version='1.0',
          title='Stock News Analyzer API',
          description='API for analyzing news sentiment for stocks')

ns = api.namespace('api', description='Stock News Analysis Operations')

# API models
stock_model = api.model('StockModel', {
    'ticker': fields.String(required=True, description='Stock ticker (e.g., AAPL)'),
    'articles': fields.Integer(default=5, description='Number of articles to analyze')
})

sentiment_model = api.model('SentimentModel', {
    'sentiment': fields.String(description='Sentiment (positive, negative, neutral)'),
    'confidence': fields.Float(description='Confidence score'),
    'scores': fields.Raw(description='Detailed sentiment scores')
})

article_model = api.model('ArticleModel', {
    'title': fields.String(description='Article title'),
    'source': fields.String(description='News source'),
    'url': fields.String(description='Article URL'),
    'text': fields.String(description='Article text'),
    'sentiment_analysis': fields.Nested(sentiment_model, description='Sentiment results'),
    'published_at': fields.String(description='Publication date')
})

response_model = api.model('ResponseModel', {
    'ticker': fields.String(description='Stock ticker'),
    'company': fields.String(description='Company name'),
    'timestamp': fields.String(description='Analysis timestamp'),
    'total_articles': fields.Integer(description='Number of articles analyzed'),
    'articles': fields.List(fields.Nested(article_model), description='Analyzed articles')
})

class FinBERTSentimentAnalyzer:
    """Handles sentiment analysis using FinBERT-tone with GPU support."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERTSentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Device selection for GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0
            logger.info("CUDA available, using GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1
            logger.info("MPS available, using Apple Silicon GPU")
        else:
            self.device = torch.device("cpu")
            device_id = -1
            logger.info("No GPU available, using CPU")

        logger.info("Loading FinBERT model...")
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model.to(self.device)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_id)
        logger.info("FinBERT model loaded successfully")
        self._initialized = True

    def batch_analyze(self, texts):
        """Analyze sentiment for a batch of texts."""
        try:
            results = self.pipeline(texts)
            processed_results = []
            for i, result in enumerate(results):
                sentiment = result['label'].lower()
                inputs = self.tokenizer(texts[i], return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                processed_results.append({
                    "sentiment": sentiment,
                    "confidence": result['score'],
                    "scores": {"positive": probs[1].item(), "negative": probs[2].item(), "neutral": probs[0].item()}
                })
            return processed_results
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return []

class StockNewsScraper:
    """Scrapes news articles and analyzes sentiment using finviz."""
    def __init__(self, ticker, max_articles=5, publish_to_rabbitmq=True):
        self.ticker = ticker.upper()
        self.company_name = self._get_company_name()
        self.max_articles = max_articles
        self.publish_to_rabbitmq = publish_to_rabbitmq
        self.finviz_url = 'https://finviz.com/quote.ashx?t='
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
        ]
        self.analyzer = FinBERTSentimentAnalyzer()
        if self.publish_to_rabbitmq:
            self.publisher = NewsPublisher(local_mode=not publish_to_rabbitmq)

    def _get_company_name(self):
        """Simple ticker-to-company mapping."""
        common_tickers = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google', 'GOOG': 'Google',
            'AMZN': 'Amazon', 'META': 'Meta', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA'
        }
        return common_tickers.get(self.ticker, self.ticker)

    def _get_random_user_agent(self):
        """Return a random user agent."""
        return random.choice(self.user_agents)

    def _scrape_finviz_news(self):
        """Scrape news from finviz for the given ticker."""
        try:
            url = self.finviz_url + self.ticker
            headers = {"User-Agent": self._get_random_user_agent()}
            
            logger.info(f"Fetching news from finviz for {self.ticker}...")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch news from finviz: status {response.status_code}")
                return []
                
            html = BeautifulSoup(response.text, features='html.parser')
            news_table = html.find(id='news-table')
            
            if not news_table:
                logger.warning(f"No news table found for {self.ticker}")
                return []
                
            parsed_data = []
            
            for row in news_table.findAll('tr')[:self.max_articles]:
                try:
                    # Extract title and URL
                    title_element = row.a
                    if not title_element:
                        continue
                    
                    title = title_element.text
                    url = title_element['href']
                    
                    # Extract date and time
                    date_data = row.td.text.split(' ')
                    
                    if len(date_data) == 1:
                        time = date_data[0]
                        # Use the date from the previous row if only time is provided
                        if not parsed_data:
                            # If this is the first row and only has time, use current date
                            date = datetime.now().strftime('%Y-%m-%d')
                        else:
                            date = parsed_data[-1]['date']
                    else:
                        date = date_data[0]
                        time = date_data[1]
                    
                    # Get the source from the URL's domain
                    source = urlparse(url).netloc
                    
                    parsed_data.append({
                        'title': title,
                        'url': url,
                        'date': date,
                        'time': time,
                        'source': source
                    })
                except Exception as e:
                    logger.error(f"Error parsing news row: {e}")
                    continue
                    
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error scraping finviz news: {e}")
            return []

    def _get_article_text(self, url):
        """Attempt to get the full text of an article using newspaper3k."""
        try:
            config = Config()
            config.browser_user_agent = self._get_random_user_agent()
            config.request_timeout = 10
            
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            if not article.text or len(article.text) < 100:
                logger.warning(f"Insufficient content for {url}")
                return None
                
            return article.text
        except Exception as e:
            logger.error(f"Error extracting article text from {url}: {e}")
            return None

    def _analyze_sentiment(self, texts):
        """Analyze sentiment for a list of texts."""
        # Filter out None values
        valid_texts = [t for t in texts if t]
        
        if not valid_texts:
            return []
            
        try:
            return self.analyzer.batch_analyze(valid_texts)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return [{"sentiment": "neutral", "confidence": 0.0, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}} for _ in valid_texts]

    def get_news(self):
        """Get news articles and analyze sentiment."""
        try:
            # Get news from finviz
            news_items = self._scrape_finviz_news()
            
            if not news_items:
                logger.warning(f"No news found for {self.ticker}")
                return []
                
            # Get full text for each article
            articles = []
            article_texts = []
            
            for item in news_items:
                text = self._get_article_text(item['url'])
                article_texts.append(text)
                
                article = {
                    'title': item['title'],
                    'url': item['url'],
                    'source': item['source'],
                    'published_at': f"{item['date']} {item['time']}",
                    'text': text if text else "No content available"
                }
                
                articles.append(article)
            
            # Analyze sentiment
            sentiments = self._analyze_sentiment(article_texts)
            
            # Match sentiments with articles
            for i, article in enumerate(articles):
                if i < len(sentiments):
                    article['sentiment_analysis'] = sentiments[i]
                else:
                    article['sentiment_analysis'] = {"sentiment": "neutral", "confidence": 0.0, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}}
            
            # Try to publish to RabbitMQ if enabled
            if self.publish_to_rabbitmq and hasattr(self, 'publisher'):
                for article in articles:
                    try:
                        sentiment = article['sentiment_analysis']['sentiment']
                        opinion = 1 if sentiment == 'positive' else (-1 if sentiment == 'negative' else 0)
                        
                        self.publisher.publish_news(
                            title=article['title'],
                            symbol=self.ticker,
                            content=article['text'],
                            published_at=datetime.strptime(article['published_at'], '%Y-%m-%d %I:%M%p') if article['published_at'] else None,
                            opinion=opinion
                        )
                    except Exception as e:
                        logger.error(f"Error publishing to RabbitMQ: {e}")
            
            return articles
            
        except Exception as e:
            logger.exception(f"Error getting news: {e}")
            return []

    def scrape_to_json(self):
        """Scrape news and return as JSON for API."""
        articles = self.get_news()
        
        result = {
            "ticker": self.ticker,
            "company": self.company_name,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "total_articles": len(articles),
            "articles": articles
        }
        
        return result

@ns.route('/analyze')
class NewsAnalyzer(Resource):
    @ns.expect(stock_model)
    @ns.marshal_with(response_model, code=200, description='News analysis with sentiment')
    def post(self):
        """Analyze news sentiment for a stock ticker."""
        data = request.json
        ticker = data.get('ticker', '').upper()
        max_articles = min(int(data.get('articles', 5)), 15)  # Limit max articles to 15

        if not ticker:
            ns.abort(400, "Ticker symbol is required")

        logger.info(f"Analyzing news for {ticker}, max articles: {max_articles}")
        
        try:
            # Create a new StockNewsScraper instance
            scraper = StockNewsScraper(ticker, max_articles=max_articles, publish_to_rabbitmq=True)
            # Call scrape_to_json directly instead of using asyncio
            result = scraper.scrape_to_json()
            return result
        except Exception as e:
            logger.exception(f"Error analyzing news for {ticker}: {e}")
            ns.abort(500, f"Error analyzing news: {str(e)}")

@ns.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

# Replace @app.before_first_request with a better approach
# Initialize models at startup
def initialize_models():
    """Initialize models on startup."""
    logger.info("Initializing sentiment analyzer...")
    FinBERTSentimentAnalyzer()
    logger.info("Sentiment analyzer initialized.")

if __name__ == "__main__":
    # Check if running locally for development (vs Docker)
    is_local = len(sys.argv) > 1 and sys.argv[1] == "--local"
    
    # Initialize models in background
    initialize_models()
    
    # Handle command line stock ticker if provided
    if len(sys.argv) > 1 and sys.argv[1] != "--local":
        ticker = sys.argv[1]
        logger.info(f"Command line analysis for ticker: {ticker}")
        scraper = StockNewsScraper(ticker, max_articles=5, publish_to_rabbitmq=not is_local)
        result = scraper.scrape_to_json()
        print(json.dumps(result, indent=2))
    else:
        # Start the Flask API server
        logger.info("Starting Flask API server...")
        app.run(debug=True, host='0.0.0.0', port=8080)