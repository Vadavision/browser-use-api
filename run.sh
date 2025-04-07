#!/bin/bash

PORT=${1:-8080}
echo "Starting browser-use microservice based on main.py on port $PORT..."
UVICORN_WORKERS=1 python -m uvicorn main:app --host 0.0.0.0 --port $PORT --reload
