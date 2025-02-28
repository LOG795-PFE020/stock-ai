#!/usr/bin/env python
# coding: utf-8

"""
Mock Web Scraping Module for Stock Sentiment Analysis

This module provides mock data for the stock sentiment analysis tool
when real web scraping is not desired or possible.
"""

import random
from typing import Dict, List, Optional

class WebScraper:
    """
    A mock web scraper that returns simulated financial news data.
    """
    
    def __init__(self, cache_dir: str = None, cache_expiry: int = 3600):
        """
        Initialize the mock web scraper.
        
        Args:
            cache_dir (str): Ignored in mock implementation
            cache_expiry (int): Ignored in mock implementation
        """
        # Company name to ticker mapping for reference
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
        if input_text.isupper() and len(input_text) <= 5:
            return input_text
        
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
    
    def _generate_mock_data(self, ticker: str, source: str, count: int = 5) -> List[str]:
        """
        Generate mock data for a specific source.
        
        Args:
            ticker (str): Stock ticker
            source (str): Source name
            count (int): Number of texts to generate
            
        Returns:
            List[str]: List of mock texts
        """
        company = self.get_company_name(ticker)
        
        templates = []
        
        if source == "YahooFinance":
            templates = [
                f"{company} beats earnings expectations, stock jumps in after-hours trading.",
                f"Why {company} (${ticker}) could be a strong buy according to analysts.",
                f"Investors worried about {company}'s growth slowdown in key markets.",
                f"{company} announces share buyback program worth billions.",
                f"Is {company} overvalued? Experts weigh in on the tech giant's prospects.",
                f"{company} CEO discusses future growth plans in exclusive interview.",
                f"New product launch could boost {company}'s revenue in Q4.",
                f"{company} faces challenges in international markets.",
                f"Analysts raise price target for {ticker} ahead of earnings.",
                f"{company} expands into new market segment with latest acquisition."
            ]
        elif source == "WSJ":
            templates = [
                f"{company} shares rose today after the company reported strong quarterly results.",
                f"Analysts are bullish on {company}'s new product lineup, raising price targets.",
                f"{company} (${ticker}) faces regulatory scrutiny over recent business practices.",
                f"Market concerns grow as {company} delays its upcoming product launch.",
                f"{company} announced a new CEO today, signaling a strategic shift.",
                f"Inside {company}'s ambitious plan to transform its business model.",
                f"How {company} is navigating supply chain disruptions.",
                f"{company} stock hits new record high amid sector rally.",
                f"Investor activism pushes {company} to consider strategic alternatives.",
                f"The tech behind {company}'s latest innovations."
            ]
        elif source == "MarketWatch":
            templates = [
                f"{company} stock rises after analyst upgrade.",
                f"Opinion: Why {company} is a buy right now.",
                f"{company} reports record revenue, but margins disappoint.",
                f"10 reasons why {company} could be the next big tech winner.",
                f"Analysts cut price targets for {company} after earnings miss.",
                f"Here's why {company} stock could continue its momentum.",
                f"Comparing {company} to its top competitors: Which is the better buy?",
                f"Economic headwinds could impact {company}'s growth trajectory.",
                f"{company} insider selling raises questions among investors.",
                f"Technical analysis suggests {ticker} is approaching key resistance level."
            ]
        elif source == "Reuters":
            templates = [
                f"{company} considering acquisitions to boost growth.",
                f"{ticker} shares fall after rival announces new competitive product.",
                f"{company} beats Wall Street forecasts, shares jump.",
                f"Exclusive: {company} in talks to acquire startup for $2 billion.",
                f"{company} CEO says company is on track for strong growth in coming year.",
                f"{company} plans to cut workforce by 5% amid restructuring.",
                f"EU regulators open investigation into {company}'s business practices.",
                f"{company} expands manufacturing capacity in Asia.",
                f"Major investor increases stake in {company} after pullback.",
                f"{company} expected to announce dividend increase next quarter."
            ]
        elif source == "Seeking Alpha":
            templates = [
                f"{company}: Buy The Dip Before Earnings",
                f"Why {company} Is Primed For A Breakout",
                f"{ticker}: Valuation Concerns Amid Slowing Growth",
                f"{company}'s Competitive Position Remains Strong Despite Challenges",
                f"Q1 Earnings Preview: What To Expect From {company}",
                f"{company} Vs. Peers: Comparative Analysis",
                f"Is {ticker} A Value Trap? Red Flags To Consider",
                f"Bull Case: Why {company} Could Double By 2026",
                f"{company}'s New Strategy Could Unlock Significant Value",
                f"Dividend Analysis: Is {company}'s Payout Sustainable?"
            ]
        
        # Ensure we don't request more templates than available
        count = min(count, len(templates))
        
        # Return a random selection of templates
        return random.sample(templates, count)
    
    def scrape_yahoo_finance(self, ticker: str) -> List[str]:
        """
        Generate mock Yahoo Finance data.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: Mock data
        """
        return self._generate_mock_data(ticker, "YahooFinance", random.randint(5, 8))
    
    def scrape_wsj(self, ticker: str) -> List[str]:
        """
        Generate mock Wall Street Journal data.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: Mock data
        """
        return self._generate_mock_data(ticker, "WSJ", random.randint(4, 7))
    
    def scrape_marketwatch(self, ticker: str) -> List[str]:
        """
        Generate mock MarketWatch data.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: Mock data
        """
        return self._generate_mock_data(ticker, "MarketWatch", random.randint(4, 7))
    
    def scrape_reuters(self, ticker: str) -> List[str]:
        """
        Generate mock Reuters data.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: Mock data
        """
        return self._generate_mock_data(ticker, "Reuters", random.randint(4, 7))
    
    def scrape_seeking_alpha(self, ticker: str) -> List[str]:
        """
        Generate mock Seeking Alpha data.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            List[str]: Mock data
        """
        return self._generate_mock_data(ticker, "Seeking Alpha", random.randint(4, 7))
    
    def scrape_all_sources(self, ticker: str) -> Dict[str, List[str]]:
        """
        Generate mock data from all sources.
        
        Args:
            ticker (str): Stock ticker or company name
            
        Returns:
            Dict[str, List[str]]: Dictionary with source names as keys and lists of texts as values
        """
        # Normalize ticker or company name
        normalized_ticker = self.normalize_ticker(ticker)
        
        # Generate mock data from each source
        return {
            "YahooFinance": self.scrape_yahoo_finance(normalized_ticker),
            "WSJ": self.scrape_wsj(normalized_ticker),
            "MarketWatch": self.scrape_marketwatch(normalized_ticker),
            "Reuters": self.scrape_reuters(normalized_ticker),
            "Seeking Alpha": self.scrape_seeking_alpha(normalized_ticker)
        }


# For testing
if __name__ == "__main__":
    scraper = WebScraper()
    ticker = "AAPL"
    
    print(f"Testing mock web scraper for {ticker}")
    results = scraper.scrape_all_sources(ticker)
    
    for source, texts in results.items():
        print(f"\n{source} ({len(texts)} items):")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")