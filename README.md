# Browser Use API

A microservice for browser automation tasks that supports real-time progress updates using Server-Sent Events (SSE).

## Overview

The Browser Use API is a FastAPI-based service that provides browser automation capabilities for the Pretuned AI platform. It handles tasks such as web scraping, form filling, and other browser-based operations, with real-time progress updates streamed back to clients.

## Features

- **Multi-agent management**: Supports multiple simultaneous browser tasks
- **Real-time updates**: Uses Server-Sent Events (SSE) to stream progress updates
- **Redis integration**: Leverages Redis for state management and task coordination
- **Health check endpoint**: Supports Kubernetes health probes
- **Containerized**: Packaged as a Docker container for easy deployment

## Deployment

The Browser Use API is deployed alongside the main NestJS backend using Kubernetes. The deployment process is automated through GitHub Actions workflows.

### Kubernetes Resources

- **Deployment**: `k8s/browser-use-deployment.yaml`
- **Service**: `k8s/browser-use-service.yaml`

## Development

### Prerequisites

- Python 3.9+
- Redis
- Playwright dependencies

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Playwright browsers: `playwright install`
4. Set up environment variables (see `.env.example`)
5. Run the service: `uvicorn main:app --reload`

## API Endpoints

### Task Management
- `POST /api/browser-use/tasks/start`: Create, start, and stream a browser task in one operation
- `GET /api/browser-use/tasks`: List all browser tasks
- `GET /api/browser-use/tasks/{task_id}`: Get information about a specific task
- `POST /api/browser-use/tasks/{task_id}/stop`: Stop a specific browser task
- `POST /api/browser-use/tasks/{task_id}/pause`: Pause a specific browser task
- `POST /api/browser-use/tasks/{task_id}/resume`: Resume a specific browser task
- `DELETE /api/browser-use/tasks/{task_id}`: Delete a specific browser task

### Debug & Health
- `GET /api/browser-use/health`: Health check endpoint for Kubernetes
- `GET /health`: Simple health check endpoint
- `GET /browser/debug-url`: Get browser's remote debugging URL for a specific task

## Integration

The Browser Use API is integrated with the main NestJS backend through the `BrowserUseService` and `BrowserUseIntentService`, which handle communication with this microservice and process the streaming updates.
