# GitHub Copilot OpenAI Adapter

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker Ready">
</p>

<p align="center">
  <b>English</b> | <a href="README_ZH.md">中文</a>
</p>

Convert GitHub Copilot API to OpenAI-compatible API format for third-party clients.

## ✨ Features

- 🔄 Full OpenAI API compatible interface
- 🌊 Streaming support (Server-Sent Events)
- 🤖 Multiple models (GPT-4o, Claude, O1, etc.)
- 🔐 OAuth Device Flow - no manual token configuration needed
- 🔑 Automatic token management and refresh
- 👥 Multi-user / multi-token support
- 🌐 CORS enabled for any client
- ⚡ High performance: HTTP/2, connection pooling, LRU cache
- 🛡️ Built-in rate limiting protection

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull from Docker Hub
docker pull h55205l/githubcp:latest

# Run
docker run -d -p 8000:8000 --name githubcp h55205l/githubcp:latest
```

### Option 2: Docker Compose

```bash
# Clone the project
git clone https://github.com/helson-lin/helson-lin-Copilot2OpenAIApI.git
cd Copilot2OpenAiApi

# Start the service
docker-compose up -d
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py

# Production (multi-process)
WORKERS=4 python server.py
# Or use gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔐 Authentication

1. Open your browser and visit `http://localhost:8000/auth/device`
2. Click the "Start Authorization" button
3. Enter the displayed code on the GitHub page and authorize
4. Get your API Key (format: `ghu_xxx`)

![OAuth Flow](https://github.githubassets.com/images/modules/device-flow/device-flow-text-logo.png)

## 📡 API Usage

### Configure Third-Party Clients

| Setting | Value |
|---------|-------|
| API Base URL | `http://localhost:8000/v1` |
| API Key | Your `ghu_xxx` key |

### Supported Clients

- ✅ [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)
- ✅ [Open WebUI](https://github.com/open-webui/open-webui)
- ✅ [Chatbox](https://github.com/Bin-Huang/chatbox)
- ✅ [BotGem](https://botgem.com/)
- ✅ [Cursor IDE](https://cursor.sh/)
- ✅ Any client supporting OpenAI API

### cURL Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ghu_xxx" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ghu_xxx"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Node.js Example

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'ghu_xxx',
});

const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.choices[0].message.content);
```

## 🤖 Available Models

Models are dynamically fetched from GitHub Copilot API. Common models include:

| Model | Description | Vision Support |
|-------|-------------|----------------|
| `gpt-4o` | GPT-4o (Recommended) | ✅ |
| `gpt-4o-mini` | GPT-4o Mini | ✅ |
| `gpt-4` | GPT-4 | ❌ |
| `gpt-3.5-turbo` | GPT-3.5 Turbo | ❌ |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet | ✅ |
| `claude-3-opus` | Claude 3 Opus | ✅ |
| `o1-preview` | O1 Preview | ❌ |
| `o1-mini` | O1 Mini | ❌ |

> 💡 Visit `/v1/models` to get the complete list of available models

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `WORKERS` | `1` | Number of worker processes |
| `HTTP_MAX_CONNECTIONS` | `100` | Maximum HTTP connections |
| `MAX_CLIENTS_CACHE` | `1000` | Client cache size |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per minute |
| `MODEL_CACHE_TTL` | `3600` | Model cache TTL (seconds) |
| `GITHUB_TOKEN` | - | Pre-configured GitHub Token |
| `GITHUB_OAUTH_CLIENT_ID` | - | Custom OAuth App ID |

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/device` | GET | OAuth device authorization page |
| `/auth/device/start` | POST | Start OAuth flow |
| `/auth/device/poll` | GET | Poll authorization status |
| `/auth/tokens` | GET | List saved tokens |
| `/auth/tokens/{api_key}` | DELETE | Delete a token |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Get model details |
| `/v1/chat/completions` | POST | Chat completions (OpenAI compatible) |
| `/health` | GET | Health check |
| `/stats` | GET | Server statistics |

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t githubcp .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v githubcp-data:/app/data \
  -e WORKERS=4 \
  -e RATE_LIMIT_REQUESTS=100 \
  --name githubcp \
  --restart unless-stopped \
  githubcp
```

### Docker Compose

```yaml
version: '3.8'
services:
  githubcp:
    image: h55205l/githubcp:latest
    ports:
      - "8000:8000"
    volumes:
      - githubcp-data:/app/data
    environment:
      - WORKERS=4
      - RATE_LIMIT_REQUESTS=100
    restart: unless-stopped

volumes:
  githubcp-data:
```

## 📈 Performance Tuning

### Recommended Configuration

| Scenario | Workers | Max Connections | Est. Concurrency |
|----------|---------|-----------------|------------------|
| Development | 1 | 100 | ~100-200 |
| Small deployment | 4 | 200 | ~400-800 |
| Production | 8+ | 500 | ~1000+ |

### Production Deployment

```bash
# Using gunicorn
gunicorn server:app \
  -w 8 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

> ⚠️ The actual bottleneck is GitHub Copilot API's own rate limiting

## 🔒 Security Recommendations

1. **Use HTTPS**: Use Nginx/Caddy with HTTPS in production
2. **Restrict Access**: Use firewall or Nginx to limit access sources
3. **Protect Tokens**: Don't expose `.github_tokens.json` file
4. **Regular Rotation**: Periodically clean up unused API keys

## ⚠️ Important Notes

1. **Copilot Subscription Required**: You need an active GitHub Copilot subscription
2. **Rate Limits**: GitHub Copilot API has usage limits, please use responsibly
3. **Compliance**: Please comply with [GitHub Copilot Terms of Service](https://docs.github.com/en/copilot/overview-of-github-copilot/about-github-copilot-individual#github-copilot-acceptable-use-policy)

## 🐛 Troubleshooting

### Authorization Failed?

- Ensure you have a valid GitHub Copilot subscription
- Check if your network can access GitHub

### Model Unavailable?

- Different Copilot subscriptions may support different models
- Try using `gpt-4o` as the default model

### Rate Limited?

- Adjust the `RATE_LIMIT_REQUESTS` environment variable
- GitHub Copilot API itself also has rate limits

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

[MIT License](LICENSE)

---

<p align="center">
  If this project helps you, please give it a ⭐️
</p>