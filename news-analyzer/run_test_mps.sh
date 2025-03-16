#!/bin/bash
echo "Building Docker image..."
docker build -t news-analyzer-mps-test .
echo "Running test_mps.py to verify MPS detection and usage..."
docker run --rm -v /tmp:/tmp -e PYTORCH_ENABLE_MPS_FALLBACK=1 news-analyzer-mps-test python test_mps.py
echo "Test complete!"
