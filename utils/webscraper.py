#!/usr/bin/env python
# coding: utf-8

"""
Web Scraping Module for Stock Sentiment Analysis

This module handles the collection of text data from financial news
and social media sources for stock sentiment analysis.
"""

import os
import re
import requests
import time
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    A web scraper for collecting financial news and social media data
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
        
        # Initialize source-specific scraping methods
        self.source_methods = {
            "YahooFinance": self.scrape_yahoo_finance,
            "WSJ": self.scrape_wsj,
            "MarketWatch": self.scrape_marketwatch,
            "Reuters": self.scrape_reuters,
            "Seeking Alpha": self.scrape_seeking_alpha,
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
        try:
            # Try to get company name from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'shortName' in info:
                return info['shortName']
            elif 'longName' in info:
                return info['longName']
        except Exception as e:
            logger.warning(f"Error getting company name for {ticker}: {e}")
        
        # Fallback to reversed mapping
        ticker_to_company = {v: k.title() for k, v in self.company_to_ticker.items()}
        return ticker_to_company.get(ticker, ticker)

    def safe_request(self, url: str, retries: int = 3, backoff_factor: float = 0.5) -> Optional[requests.Response]:
        """
        Make a safe HTTP request with retries and backoff.
        
        Args:
            url (str): URL to request
            retries (int): Number of retries
            backoff_factor (float): Backoff factor
            
        Returns:
            Optional[requests.Response]: Response object or None if all retries failed
        """
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
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
    
    def scrape_yahoo_finance(self, ticker: str) -> List[str]:
        """
        Scrape Yahoo Finance for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "YahooFinance")
        if cached_data:
            return cached_data
        
        texts = []
        
        # Get news from Yahoo Finance
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            for item in news:
                if 'title' in item:
                    texts.append(item['title'])
                if 'summary' in item:
                    texts.append(item['summary'])
        except Exception as e:
            logger.warning(f"Error getting Yahoo Finance news for {ticker}: {e}")
        
        # Get more news from the Yahoo Finance website
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines
            headlines = soup.select('h3.Mb\\(5px\\)')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Find news summaries
            summaries = soup.select('p.Fz\\(14px\\)')
            for summary in summaries:
                if summary.text and len(summary.text.strip()) > 20:
                    texts.append(summary.text.strip())
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "YahooFinance", texts)
        
        return texts
    
    def scrape_wsj(self, ticker: str) -> List[str]:
        """
        Scrape Wall Street Journal for news related to a ticker.
        Note: WSJ has a paywall, so we can only get limited content.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "WSJ")
        if cached_data:
            return cached_data
        
        texts = []
        company_name = self.get_company_name(ticker)
        
        # Search for the company in WSJ
        url = f"https://www.wsj.com/search?query={company_name}&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2020%2F02%2F26&endDate=2025%2F02%2F26&source=wsjie%2Cblog%2Cwsjsitesrch%2Cwsjpro%2Cautowire%2Capfeed"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find article headlines
            headlines = soup.select('.WSJTheme--headline--7VCzo7Ay')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Try to find article summaries
            summaries = soup.select('.WSJTheme--snippet--2bO1LgJu')
            for summary in summaries:
                if summary.text and len(summary.text.strip()) > 20:
                    texts.append(summary.text.strip())
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "WSJ", texts)
        
        return texts
    
    def scrape_marketwatch(self, ticker: str) -> List[str]:
        """
        Scrape MarketWatch for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "MarketWatch")
        if cached_data:
            return cached_data
        
        texts = []
        
        # Get news from MarketWatch
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines in the Latest News section
            headlines = soup.select('.article__headline')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Find other related headlines
            other_headlines = soup.select('.link')
            for headline in other_headlines:
                text = headline.text.strip()
                if text and len(text) > 10 and text not in texts:
                    texts.append(text)
        
        # Get more from the news page
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/news"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines
            headlines = soup.select('.headline')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Find news summaries
            summaries = soup.select('.description')
            for summary in summaries:
                if summary.text and len(summary.text.strip()) > 20:
                    texts.append(summary.text.strip())
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "MarketWatch", texts)
        
        return texts
    
    def scrape_reuters(self, ticker: str) -> List[str]:
        """
        Scrape Reuters for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "Reuters")
        if cached_data:
            return cached_data
        
        texts = []
        
        # Try to guess the Reuters stock URL format
        url = f"https://www.reuters.com/markets/companies/{ticker}/"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines
            headlines = soup.select('.text__text__1FZLe.text__dark-grey__3Ml43.text__medium__1kbOh.hover-highlight')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Find other news items
            other_headlines = soup.select('a[data-testid="Heading"]')
            for headline in other_headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
        
        # Try to search Reuters for the company name
        company_name = self.get_company_name(ticker)
        url = f"https://www.reuters.com/search/?blob={company_name}"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find search result headlines
            headlines = soup.select('.search-result-title')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "Reuters", texts)
        
        return texts
    
    def scrape_seeking_alpha(self, ticker: str) -> List[str]:
        """
        Scrape Seeking Alpha for news related to a ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: List of text content
        """
        # Check cache first
        cached_data = self._read_cache(ticker, "Seeking Alpha")
        if cached_data:
            return cached_data
        
        texts = []
        
        # Get news from Seeking Alpha
        url = f"https://seekingalpha.com/symbol/{ticker}"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines
            headlines = soup.select('a.sasq-189')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
        
        # Get more from the news tab
        url = f"https://seekingalpha.com/symbol/{ticker}/news"
        response = self.safe_request(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news headlines
            headlines = soup.select('a[data-test-id="post-list-item-title"]')
            for headline in headlines:
                if headline.text and len(headline.text.strip()) > 10:
                    texts.append(headline.text.strip())
            
            # Find news summaries
            summaries = soup.select('div[data-test-id="post-list-item-summary"]')
            for summary in summaries:
                if summary.text and len(summary.text.strip()) > 20:
                    texts.append(summary.text.strip())
        
        # Cache the results
        if texts:
            self._write_cache(ticker, "Seeking Alpha", texts)
        
        return texts
    
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
        
        # Scrape each source
        for source, method in self.source_methods.items():
            try:
                logger.info(f"Scraping {source} for {normalized_ticker}")
                texts = method(normalized_ticker)
                
                if texts:
                    # Remove duplicates while preserving order
                    unique_texts = []
                    seen = set()
                    for text in texts:
                        if text not in seen:
                            unique_texts.append(text)
                            seen.add(text)
                    
                    results[source] = unique_texts
                    logger.info(f"Found {len(unique_texts)} unique texts from {source}")
                else:
                    logger.info(f"No texts found from {source}")
            except Exception as e:
                logger.error(f"Error scraping {source} for {normalized_ticker}: {e}")
        
        # Check if we have any results
        if not results:
            logger.warning(f"No news found for {normalized_ticker} from any source")
        
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