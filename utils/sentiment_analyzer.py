from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch


class FinBERTSentimentAnalyzer:
    def __init__(self):
        """Initialize the FinBERT sentiment analyzer with pre-trained model and tokenizer."""

                # Device priority: CUDA GPU > Apple MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0  # Use first CUDA device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1  # MPS is not supported directly in pipeline, will move manually
        else:
            self.device = torch.device("cpu")
            device_id = -1
            
        print(f"Using device: {self.device}")
        
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Initialize pipeline with appropriate device
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer,
                               device=device_id)
        self.label_map = {"neutral": 0, "positive": 1, "negative": 2}  # Correct label mapping
    
    def analyze(self, text):
        """
        Analyze the sentiment of a single text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: A dictionary containing sentiment, confidence, and detailed scores
        """
        result = self.pipeline(text)[0]
        sentiment = result['label'].lower()
        
        # Move inputs to same device as model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        return {
            "sentiment": sentiment,
            "confidence": result['score'],
            "scores": {
                "positive": probs[1].item(),
                "negative": probs[2].item(),
                "neutral": probs[0].item()
            }
        }
    
    def batch_analyze(self, texts):
        """
        Analyze the sentiment of multiple texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of dictionaries containing sentiment analysis results
        """
        results = self.pipeline(texts)
        processed_results = []
        
        for i, result in enumerate(results):
            sentiment = result['label'].lower()
            
            inputs = self.tokenizer(texts[i], return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
            processed_results.append({
                "sentiment": sentiment,
                "confidence": result['score'],
                "scores": {
                    "positive": probs[1].item(),
                    "negative": probs[2].item(),
                    "neutral": probs[0].item()
                }
            })
            
        return processed_results


# Example usage
if __name__ == "__main__":
    analyzer = FinBERTSentimentAnalyzer()
    
    # Single text analysis
    text = "The company reported strong earnings, exceeding market expectations."
    result = analyzer.analyze(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Scores: {result['scores']}")
    
    # Batch analysis
    texts = [
        "There is a shortage of capital, and we need extra financing",  
        "Growth is strong and we have plenty of liquidity", 
        "There are doubts about our finances", 
        "Profits are flat"
    ]
    results = analyzer.batch_analyze(texts)
    
    print("\nBatch Analysis Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Text: {texts[i]}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.4f}")

