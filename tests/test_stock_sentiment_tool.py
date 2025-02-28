import unittest
from unittest.mock import patch, MagicMock
import json
import re
from datetime import datetime

# Import our stock sentiment tool
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_sentiment_tool import StockSentimentTool, MockSentimentAnalyzer


class TestMockSentimentAnalyzer(unittest.TestCase):
    """Test the mock sentiment analyzer."""
    
    def setUp(self):
        """Set up the test environment."""
        self.analyzer = MockSentimentAnalyzer()
        
    def test_batch_analyze_structure(self):
        """Test the structure of batch_analyze results."""
        texts = ["Some positive text", "Some negative text"]
        results = self.analyzer.batch_analyze(texts)
        
        # Check results structure
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIn("sentiment", result)
            self.assertIn("confidence", result)
            self.assertIn("scores", result)
            self.assertIn("positive", result["scores"])
            self.assertIn("negative", result["scores"])
            self.assertIn("neutral", result["scores"])
            
            # Check confidence value range
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            
            # Check scores sum to approximately 1
            total_score = sum(result["scores"].values())
            self.assertAlmostEqual(total_score, 1.0, places=1)
            
    def test_sentiment_classification(self):
        """Test sentiment classification based on keywords."""
        positive_text = "The company is bullish and stock rose strong"
        negative_text = "There are concerns and people are worried about scrutiny"
        
        results = self.analyzer.batch_analyze([positive_text, negative_text])
        
        # Positive text should have higher positive score
        self.assertGreater(results[0]["scores"]["positive"], results[0]["scores"]["negative"])
        
        # Negative text should have higher negative score
        self.assertGreater(results[1]["scores"]["negative"], results[1]["scores"]["positive"])


class TestStockSentimentTool(unittest.TestCase):
    """Test the stock sentiment tool."""
    
    def setUp(self):
        """Set up the test environment."""
        # Use the mock analyzer for testing
        self.sentiment_tool = StockSentimentTool(use_mock=True)
        
    def test_get_mock_data(self):
        """Test the get_mock_data method."""
        ticker = "AAPL"
        mock_data = self.sentiment_tool.get_mock_data(ticker)
        
        # Check structure of mock data
        expected_sources = ["WallStreet Journal", "YahooFinance", "Reddit", "Twitter"]
        self.assertEqual(set(mock_data.keys()), set(expected_sources))
        
        # Check that each source has texts
        for source, texts in mock_data.items():
            self.assertGreater(len(texts), 0)
            
        # Check that ticker is present in texts
        all_texts = []
        for texts in mock_data.values():
            all_texts.extend(texts)
            
        ticker_found = False
        for text in all_texts:
            if ticker in text or ticker.lower() in text.lower():
                ticker_found = True
                break
                
        self.assertTrue(ticker_found, f"Ticker {ticker} not found in any text")
        
    def test_calculate_relevance(self):
        """Test the calculate_relevance method."""
        ticker = "AAPL"
        
        # Test with text containing the ticker
        texts_with_ticker = [
            "AAPL stock is doing well",
            "$AAPL is a good investment",
            "I like #AAPL products"
        ]
        relevance1 = self.sentiment_tool.calculate_relevance(texts_with_ticker, ticker)
        self.assertGreaterEqual(relevance1, 0.0)
        self.assertLessEqual(relevance1, 1.0)
        
        # Test with text not containing the ticker
        texts_without_ticker = [
            "The stock market is volatile",
            "Technology companies are facing challenges",
            "Investors are looking for stable returns"
        ]
        relevance2 = self.sentiment_tool.calculate_relevance(texts_without_ticker, ticker)
        self.assertGreaterEqual(relevance2, 0.0)
        self.assertLessEqual(relevance2, 1.0)
        
        # Test with empty list
        relevance3 = self.sentiment_tool.calculate_relevance([], ticker)
        self.assertEqual(relevance3, 0.0)
        
    def test_analyze_sentiment(self):
        """Test the analyze_sentiment method."""
        ticker = "AAPL"
        
        # Mock get_mock_data to return controlled data
        with patch.object(self.sentiment_tool, 'get_mock_data') as mock_get_data:
            mock_get_data.return_value = {
                "WallStreet Journal": ["AAPL stock rose today"],
                "YahooFinance": ["AAPL facing concerns"],
                "Reddit": ["AAPL to the moon!"],
                "Twitter": ["$AAPL is stable"]
            }
            
            results = self.sentiment_tool.analyze_sentiment(ticker)
            
            # Check results structure
            self.assertEqual(results["ticker"], ticker)
            self.assertIn(results["sentiment"], ["positive", "negative", "neutral"])
            self.assertGreaterEqual(results["relevance"], 0.0)
            self.assertLessEqual(results["relevance"], 1.0)
            self.assertEqual(len(results["source_breakdown"]), 4)
            
            # Check date format
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            self.assertTrue(re.match(date_pattern, results["date"]))
            
    def test_get_sentiment_json(self):
        """Test the get_sentiment_json method."""
        ticker = "AAPL"
        
        # Mock analyze_sentiment to return controlled data
        with patch.object(self.sentiment_tool, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                "ticker": ticker,
                "sentiment": "positive",
                "relevance": 0.85,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source_breakdown": [],
                "model": "mock"
            }
            
            json_result = self.sentiment_tool.get_sentiment_json(ticker)
            
            # Parse JSON result
            parsed_result = json.loads(json_result)
            
            # Check JSON structure
            self.assertEqual(parsed_result["sentiment"], "positive")
            self.assertEqual(parsed_result["relevance"], 0.85)
            self.assertIn("date", parsed_result)


if __name__ == '__main__':
    unittest.main()