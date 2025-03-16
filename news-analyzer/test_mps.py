#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify MPS (Apple Silicon GPU) detection and usage
"""

import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mps_availability():
    """Test MPS availability and setup a simple model to verify it works"""
    logger.info("Testing MPS availability on this system")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_id = 0
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_id = "mps"
        logger.info("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        device_id = -1
        logger.info("Using CPU")
    
    logger.info(f"Device selected: {device}")
    
    # Create a simple tensor and verify it's on the right device
    x = torch.randn(10, 10)
    x = x.to(device)
    logger.info(f"Tensor device: {x.device}")
    
    # Try to load and run a small model
    logger.info("Loading FinBERT model...")
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model.to(device)
    
    # Test inference
    test_text = "The company reported strong earnings growth."
    logger.info(f"Running inference on test text: '{test_text}'")
    
    # Encode input
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        import time
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
        
        inference_time = (end - start) * 1000  # Convert to milliseconds
    
    # Get sentiment
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    sentiment_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    sentiment_idx = torch.argmax(probs).item()
    sentiment = sentiment_map[sentiment_idx]
    
    logger.info(f"Inference took {inference_time:.2f} ms")
    logger.info(f"Sentiment: {sentiment}")
    logger.info(f"Probabilities: neutral={probs[0]:.4f}, positive={probs[1]:.4f}, negative={probs[2]:.4f}")
    
    logger.info("MPS test completed successfully!")

if __name__ == "__main__":
    test_mps_availability() 