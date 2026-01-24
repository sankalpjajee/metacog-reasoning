#!/bin/bash
# Start MLflow UI to view experiment tracking

PORT=${1:-5000}

echo "Starting MLflow UI on port $PORT..."
echo "Access at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"

mlflow ui --port $PORT --backend-store-uri file:./mlruns
