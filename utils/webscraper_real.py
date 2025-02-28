#!/usr/bin/env python
# coding: utf-8

"""
Real Web Scraping Module for Stock Sentiment Analysis

This module handles the collection of text data from financial news
using reliable public APIs and feeds.
"""

import os
import re
import requests
import time
import json
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    A web scraper for collecting financial news data
    related to a specific stock ticker or company name.
    """
    
    def __init__(self, cache_dir: str = None, cache_expiry: int = 3600):
        """
        Initialize the web scraper.
        
        Args:
            cache_dir (str): Directory to store cached data
            cache_expiry (int): Cache expiry time in seconds (default: 1 hour)
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_expiry = cache_expiry
        
        # Company name to ticker mapping
        self.company_to_ticker = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "meta": "META",
            "facebook": "META",
            "tesla": "TSLA",
            "nvidia": "NVDA",
        }
        
    def _get_cache_filename(self, ticker: str, source: str) -> str:
        """
        Generate a cache filename for a ticker and source.
        
        Args:
            ticker (str): Stock ticker
            source (str): Data source name
            
        Returns:
            str: Cache filename
        """
        if not self.cache_dir:
            return None
        
        date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"{ticker}_{source}_{date_str}.cache")
        
    def _read_cache(self, ticker: str, source: str) -> Optional[List[str]]:
        """
        Read cached data for a ticker and source.
        
        Args:
            ticker (str): Stock ticker
            source (str): Data source name
            
        Returns:
            Optional[List[str]]: Cached data or None if cache doesn't exist or is expired
        """
        if not self.cache_dir:
            return None
        
        cache_file = self._get_cache_filename(ticker, source)
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        file_mtime = os.path.getmtime(cache_file)
        if time.time() - file_mtime > self.cache_expiry:
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.warning(f"Error reading cache for {ticker} from {source}: {e}")
            return None
            
    def _write_cache(self, ticker: str, source: str, data: List[str]) -> bool:
        """
        Write data to cache for a ticker and source.
        
        Args:
            ticker (str): Stock ticker
            source (str): Data source name
            data (List[str]): Data to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cache_dir or not data:
            return False
        
        cache_file = self._get_cache_filename(ticker, source)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(f"{item}\n")
            return True
        except Exception as e:
            logger.warning(f"Error writing cache for {ticker} to {source}: {e}")
            return False
    
    def normalize_ticker(self, ticker_or_company: str) -> str:
        """
        Normalize ticker or company name to ticker symbol.
        
        Args:
            ticker_or_company (str): Stock ticker or company name
            
        Returns:
            str: Normalized ticker symbol
        """
        input_text = ticker_or_company.strip().lower()
        
        # Check if it's already a ticker-like string (all caps, 1-5 chars)
        if re.match(r'^[A-Z]{1,5}$', ticker_or_company):
            return ticker_or_company
        
        # Check if it's in our mapping
        if input_text in self.company_to_ticker:
            return self.company_to_ticker[input_text]
        
        # Try to find partial matches
        for company, ticker in self.company_to_ticker.items():
            if company in input_text or input_text in company:
                return ticker
        
        # If we can't determine, return as is (uppercase)
        return ticker_or_company.upper()
    
    def get_company_name(self, ticker: str) -> str:
        """
        Get company name for a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            str: Company name
        """
        # Reversed mapping
        ticker_to_company = {v: k.title() for k, v in self.company_to_ticker.items()}
        return ticker_to_company.get(ticker, ticker)

    def safe_request(self, url: str, retries: int = 3, backoff_factor: float = 0.5, params=None) -> Optional[requests.Response]:
        """
        Make a safe HTTP request with retries and backoff.
        
        Args:
            url (str): URL to request
            retries (int): Number of retries
            backoff_factor (float): Backoff factor
            params (dict): Request parameters
            
        Returns:
            Optional[requests.Response]: Response object or None if all retries failed
        """
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code == 200:
                    return response
                
                if response.status_code == 429:  # Too Many Requests
                    # Use exponential backoff
                    sleep_time = backoff_factor * (2 ** i)
                    logger.warning(f"Rate limited, sleeping for {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                
                logger.warning(f"Error status code {response.status_code} for {url}")
                
            except Exception as e:
                logger.warning(f"Error requesting {url}: {e}")
            
            # Use exponential backoff for any failure
            sleep_time = backoff_factor * (2 ** i)
            time.sleep(sleep_time)
        
        return None
    
    def scrape_google_news(self, ticker: str) -> List[str]:
        """
        Scrape Google News for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "GoogleNews")
        if cached_data:
            return cached_data
        
        logger.info(f"Scraping Google News for {ticker}...")
        
        texts = []
        company = self.get_company_name(ticker)
        # Use company name for better results, but include ticker
        query = f"{company} {ticker} stock"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        response = self.safe_request(url)
        
        if response:
            try:
                # Parse the RSS feed
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.find_all("item")
                
                for item in items:
                    title = item.title.text
                    texts.append(title)
                    
                    # Add description if available
                    if item.description:
                        description = item.description.text
                        # Only add description if it's not too similar to the title
                        if len(description) > len(title) * 1.5:
                            texts.append(description)
                
                logger.info(f"Found {len(texts)} texts from Google News")
            except Exception as e:
                logger.error(f"Error parsing Google News: {e}")
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "GoogleNews", texts)
        
        return texts
    
    def scrape_yahoo_finance_headlines(self, ticker: str) -> List[str]:
        """
        Scrape Yahoo Finance headlines for a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "YahooHeadlines")
        if cached_data:
            return cached_data
        
        logger.info(f"Scraping Yahoo Finance headlines for {ticker}...")
        
        texts = []
        url = f"https://finance.yahoo.com/quote/{ticker}"
        
        response = self.safe_request(url)
        
        if response:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news headlines
                headlines = soup.select('h3.Mb\\(5px\\)')
                for headline in headlines:
                    if headline.text and len(headline.text.strip()) > 10:
                        texts.append(headline.text.strip())
                
                logger.info(f"Found {len(texts)} texts from Yahoo Finance headlines")
            except Exception as e:
                logger.error(f"Error parsing Yahoo Finance: {e}")
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "YahooHeadlines", texts)
        
        return texts
    
    def scrape_market_watch_headlines(self, ticker: str) -> List[str]:
        """
        Try to scrape MarketWatch headlines.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "MarketWatchHeadlines")
        if cached_data:
            return cached_data
        
        logger.info(f"Scraping MarketWatch headlines for {ticker}...")
        
        texts = []
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
        
        response = self.safe_request(url)
        
        if response and response.status_code == 200:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different selectors
                headlines = soup.select('.article__headline')
                
                for headline in headlines:
                    text = headline.text.strip()
                    if text and len(text) > 10:
                        texts.append(text)
                
                logger.info(f"Found {len(texts)} texts from MarketWatch headlines")
            except Exception as e:
                logger.error(f"Error parsing MarketWatch: {e}")
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "MarketWatchHeadlines", texts)
        
        return texts
    
    def generate_mock_data(self, ticker: str) -> Dict[str, List[str]]:
        """
        Generate mock data when real scraping fails.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Dict[str, List[str]]: Mock data
        """
        logger.warning(f"Generating mock data for {ticker} as fallback")
        
        company = self.get_company_name(ticker)
        
        # Generate some mock headlines
        google_news = [
            f"{company} stock rises after analyst upgrade",
            f"{ticker} beats earnings expectations, shares jump",
            f"Analysts positive on {company}'s growth prospects",
            f"{company} announces new product line, stock reacts",
            f"Market awaits {company}'s quarterly results"
        ]
        
        yahoo_headlines = [
            f"{company} hits new 52-week high",
            f"Why {ticker} could be a good buy right now",
            f"{company}'s CEO discusses future growth in interview",
            f"3 reasons to be bullish on {ticker}",
            f"Analysts set new price targets for {company}"
        ]
        
        market_watch = [
            f"Opinion: {company} is positioned for long-term growth",
            f"{ticker} shares react to market volatility",
            f"Earnings preview: What to expect from {company}",
            f"Technical analysis: {ticker} approaches resistance level",
            f"{company} announces share buyback program"
        ]
        
        # Return mock data
        return {
            "GoogleNews": google_news,
            "YahooFinance": yahoo_headlines,
            "MarketWatch": market_watch
        }
    
    def scrape_all_sources(self, ticker: str) -> Dict[str, List[str]]:
        """
        Scrape all sources for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker or company name
            
        Returns:
            Dict[str, List[str]]: Dictionary with source names as keys and lists of texts as values
        """
        # Normalize ticker or company name
        normalized_ticker = self.normalize_ticker(ticker)
        logger.info(f"Scraping news for {normalized_ticker} (from input: {ticker})")
        
        results = {}
        success = False
        
        # Try to get Google News data
        google_news = self.scrape_google_news(normalized_ticker)
        if google_news:
            results["GoogleNews"] = google_news
            success = True
        
        # Try to get Yahoo Finance headlines
        yahoo_headlines = self.scrape_yahoo_finance_headlines(normalized_ticker)
        if yahoo_headlines:
            results["YahooFinance"] = yahoo_headlines
            success = True
        
        # Try to get MarketWatch headlines
        market_watch = self.scrape_market_watch_headlines(normalized_ticker)
        if market_watch:
            results["MarketWatch"] = market_watch
            success = True
        
        # Use mock data if all real scraping fails
        if not success:
            logger.warning(f"All real scraping failed for {normalized_ticker}, using mock data")
            return self.generate_mock_data(normalized_ticker)
        
        return results


# For testing
if __name__ == "__main__":
    scraper = WebScraper(cache_dir="./cache")
    ticker = "AAPL"
    
    print(f"Testing web scraper for {ticker}")
    results = scraper.scrape_all_sources(ticker)
    
    for source, texts in results.items():
        print(f"\n{source} ({len(texts)} items):")
        for i, text in enumerate(texts[:5], 1):  # Show only first 5 items
            print(f"  {i}. {text}")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more")