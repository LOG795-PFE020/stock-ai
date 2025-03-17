#!/bin/bash

# Set your DockerHub username
DOCKERHUB_USERNAME="hyukay"

# Enable Docker BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Create the auth network if it doesn't exist
docker network inspect auth >/dev/null 2>&1 || docker network create auth

# Build and push the news-analyzer image
echo "Building news-analyzer image..."
docker build --tag $DOCKERHUB_USERNAME/news-analyzer:latest \
             --cache-from $DOCKERHUB_USERNAME/news-analyzer:latest \
             --build-arg BUILDKIT_INLINE_CACHE=1 \
             ./news-analyzer

echo "Pushing news-analyzer image to DockerHub..."
docker push $DOCKERHUB_USERNAME/news-analyzer:latest

# Build and push the stock-predictor image
echo "Building stock-predictor image..."
docker build --tag $DOCKERHUB_USERNAME/stock-predictor:latest \
             --cache-from $DOCKERHUB_USERNAME/stock-predictor:latest \
             --build-arg BUILDKIT_INLINE_CACHE=1 \
             .

echo "Pushing stock-predictor image to DockerHub..."
docker push $DOCKERHUB_USERNAME/stock-predictor:latest

echo "Build and push complete!"
echo ""
echo "To start the services, run:"
echo "docker-compose up -d"
echo ""
echo "To check the logs:"
echo "docker-compose logs -f" 