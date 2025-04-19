# Browser Use API

A microservice for browser automation tasks that supports real-time progress updates using Server-Sent Events (SSE).

## Overview

The Browser Use API is a FastAPI-based service that provides browser automation capabilities for the Pretuned AI platform. It handles tasks such as web scraping, form filling, and other browser-based operations, with real-time progress updates streamed back to clients.

## Features

- **Multi-agent management**: Supports multiple simultaneous browser tasks
- **Real-time updates**: Uses Server-Sent Events (SSE) to stream progress updates
- **Redis integration**: Leverages Redis for state management and task coordination
- **S3 screenshot storage**: Optionally stores screenshots in S3 and returns URLs instead of base64 data
- **Enhanced stealth capabilities**: Integrates with patchright for improved bot detection avoidance
- **Cloud browser support**: Can connect to external cloud browsers for enhanced reliability and stealth
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

### Environment Variables

The following environment variables can be configured:

```
# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379

# AWS S3 Configuration (Optional)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-screenshot-bucket
S3_PREFIX=screenshots

# Cloud Browser Configuration (Optional)
CLOUD_BROWSER_URL=wss://your-cloud-browser-provider.com/browser/ws-endpoint

# Proxy Configuration (Optional)
USE_PROXY=true
PROXY_SERVER=geo.iproyal.com:12321
PROXY_USERNAME=your_proxy_username
PROXY_PASSWORD=your_proxy_password
```

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

## S3 Screenshot Storage

The API can optionally store screenshots in Amazon S3 instead of including them directly in the response payload. This significantly reduces the size of the streamed data and improves performance.

### Configuration

To enable S3 screenshot storage, set the following environment variables:

```
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-screenshot-bucket
S3_PREFIX=screenshots  # Folder prefix within the bucket
```

### How It Works

1. When S3 is properly configured, screenshots are uploaded to the specified S3 bucket
2. The API returns a `screenshot_url` field instead of the `screenshot` field in the state updates
3. If S3 upload fails or is not configured, the API falls back to including the raw screenshot data

## Cloud Browser Integration

The API supports connecting to external cloud browsers instead of launching local browsers. This provides several benefits:

1. **Enhanced stealth**: Cloud browsers often have better fingerprinting protection
2. **Improved reliability**: Avoid local browser installation and compatibility issues
3. **Distributed IP addresses**: Cloud browsers can provide access from different geographic locations
4. **Resource efficiency**: Offload browser processing to external services

### Configuration

To use a cloud browser, set the following environment variable:

```
CLOUD_BROWSER_URL=wss://your-cloud-browser-provider.com/browser/ws-endpoint
```

The URL can be either:
- A WebSocket URL (starting with `wss://`)
- A Chrome DevTools Protocol URL (typically `http://hostname:port`)

### Recommended Cloud Browser Providers

- **BrowserStack**: Enterprise-grade cloud browser platform
- **Browserless.io**: Purpose-built for automation
- **Browser-use Cloud**: Native integration with browser-use
- **Bright Data's Unblocker**: Specialized anti-detection cloud browser

### Testing Cloud Browser Integration

Use the included test script to verify cloud browser functionality:

```bash
# Set the cloud browser URL
export CLOUD_BROWSER_URL="wss://your-cloud-browser-provider.com/browser/ws-endpoint"

# Run the test script
python test_cloud_browser.py
```

## Proxy Integration

The Browser Use API supports connecting through proxy servers to enhance anonymity and avoid IP-based blocking. This is particularly useful for bot detection avoidance and accessing geo-restricted content.

### Configuration

To use a proxy server, set the following environment variables:

```
USE_PROXY=true
PROXY_SERVER=geo.iproyal.com:12321  # Proxy server address and port
PROXY_USERNAME=your_proxy_username  # Proxy authentication username
PROXY_PASSWORD=your_proxy_password  # Proxy authentication password
```

### Supported Proxy Providers

- **IPRoyal**: Residential and datacenter proxies with geo-targeting
- **Bright Data**: Enterprise-grade proxy network
- **Oxylabs**: Residential and datacenter proxies
- **Any HTTP proxy**: Standard HTTP proxies with authentication

## Enhanced Stealth Capabilities

The API integrates with patchright to provide enhanced stealth capabilities, making browser automation less detectable. Key features include:

- **WebDriver fingerprint masking**: Prevents detection of automation frameworks
- **Browser fingerprint normalization**: Makes automated browsers appear more like regular users
- **User agent spoofing**: Uses realistic user agents that match browser versions
- **Hardware fingerprint simulation**: Simulates realistic hardware characteristics
- **Timezone and locale settings**: Configures browser to use consistent geographic settings

These stealth capabilities significantly improve success rates when interacting with websites that employ bot detection mechanisms.

## Integration

The Browser Use API is integrated with the main NestJS backend through the `BrowserUseService` and `BrowserUseIntentService`, which handle communication with this microservice and process the streaming updates.
