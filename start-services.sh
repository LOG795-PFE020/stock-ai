#!/bin/bash

# Create the auth network if it doesn't exist
docker network inspect auth >/dev/null 2>&1 || docker network create auth

# Pull images from DockerHub if they don't exist locally
if [[ "$(docker images -q hyukay/news-analyzer:latest 2> /dev/null)" == "" ]]; then
  echo "Pulling news-analyzer image from DockerHub..."
  docker pull hyukay/news-analyzer:latest
fi

if [[ "$(docker images -q hyukay/stock-predictor:latest 2> /dev/null)" == "" ]]; then
  echo "Pulling stock-predictor image from DockerHub..."
  docker pull hyukay/stock-predictor:latest
fi

# Start the services
echo "Starting services..."
docker compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Check if services are running
if docker compose ps | grep -q "Up"; then
  echo "Services are running!"
  echo ""
  echo "News Analyzer API is available at: http://localhost:8092"
  echo "Stock Predictor API is available at: http://localhost:8000"
  echo "RabbitMQ Management UI is available at: http://localhost:15672 (guest/guest)"
  echo ""
  echo "To view logs: docker-compose logs -f"
  echo "To stop services: docker-compose down"
else
  echo "Error: Some services failed to start. Check logs:"
  docker compose logs
fi 