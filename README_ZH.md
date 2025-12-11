# GitHub Copilot OpenAI Adapter

[English](README.md) | **中文**

将 GitHub Copilot API 转换为 OpenAI 兼容的 API 格式，供第三方客户端调用。

## ✨ 功能特性

- 🔄 完整的 OpenAI API 兼容接口
- 🌊 支持流式输出 (Server-Sent Events)
- 🤖 支持多种模型 (GPT-4o, Claude, O1 等)
- 🔐 OAuth 设备流自动授权，无需手动配置 Token
- 🔑 自动 Token 管理和刷新
- 👥 支持多用户/多 Token
- 🌐 CORS 支持，可被任何客户端调用
- ⚡ 高性能：HTTP/2、连接池、LRU 缓存
- 🛡️ 内置速率限制保护

## 🚀 快速开始

### 方式一：Docker（推荐）

```bash
# 从 Docker Hub 拉取
docker pull h55205l/githubcp:latest

# 运行
docker run -d -p 8000:8000 --name githubcp h55205l/githubcp:latest
```

### 方式二：Docker Compose

```bash
# 克隆项目
git clone https://github.com/helson-lin/helson-lin-Copilot2OpenAIApI.git
cd Copilot2OpenAiApi

# 启动服务
docker-compose up -d
```

### 方式三：本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python server.py

# 生产环境（多进程）
WORKERS=4 python server.py
# 或
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔐 认证授权

1. 打开浏览器访问 `http://localhost:8000/auth/device`
2. 点击「开始授权」按钮
3. 在 GitHub 页面输入显示的代码并授权
4. 获取 API Key（格式：`ghu_xxx`）

![OAuth Flow](https://github.githubassets.com/images/modules/device-flow/device-flow-text-logo.png)

## 📡 API 使用

### 配置第三方客户端

| 配置项 | 值 |
|--------|-----|
| API Base URL | `http://localhost:8000/v1` |
| API Key | 你获取的 `ghu_xxx` |

### 支持的客户端

- ✅ [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)
- ✅ [Open WebUI](https://github.com/open-webui/open-webui)
- ✅ [Chatbox](https://github.com/Bin-Huang/chatbox)
- ✅ [BotGem](https://botgem.com/)
- ✅ [Cursor IDE](https://cursor.sh/)
- ✅ 任何支持 OpenAI API 的客户端

### cURL 示例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ghu_xxx" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好！"}],
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
    messages=[{"role": "user", "content": "你好！"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Node.js 示例

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'ghu_xxx',
});

const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '你好！' }],
});

console.log(response.choices[0].message.content);
```

## 🤖 可用模型

模型列表从 GitHub Copilot API 动态获取，常见模型包括：

| 模型 | 描述 | 支持视觉 |
|------|------|---------|
| `gpt-4o` | GPT-4o（推荐） | ✅ |
| `gpt-4o-mini` | GPT-4o Mini | ✅ |
| `gpt-4` | GPT-4 | ❌ |
| `gpt-3.5-turbo` | GPT-3.5 Turbo | ❌ |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet | ✅ |
| `claude-3-opus` | Claude 3 Opus | ✅ |
| `o1-preview` | O1 Preview | ❌ |
| `o1-mini` | O1 Mini | ❌ |

> 💡 访问 `/v1/models` 获取完整的可用模型列表

## ⚙️ 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `WORKERS` | `1` | 工作进程数 |
| `HTTP_MAX_CONNECTIONS` | `100` | 最大 HTTP 连接数 |
| `MAX_CLIENTS_CACHE` | `1000` | 客户端缓存大小 |
| `RATE_LIMIT_ENABLED` | `true` | 是否启用速率限制 |
| `RATE_LIMIT_REQUESTS` | `60` | 每分钟最大请求数 |
| `MODEL_CACHE_TTL` | `3600` | 模型缓存时间（秒） |
| `GITHUB_TOKEN` | - | 预配置的 GitHub Token |
| `GITHUB_OAUTH_CLIENT_ID` | - | 自定义 OAuth App ID |

## 📊 API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/auth/device` | GET | OAuth 设备授权页面 |
| `/auth/device/start` | POST | 开始 OAuth 流程 |
| `/auth/device/poll` | GET | 轮询授权状态 |
| `/auth/tokens` | GET | 列出已保存的 Token |
| `/auth/tokens/{api_key}` | DELETE | 删除指定 Token |
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/models/{model_id}` | GET | 获取模型详情 |
| `/v1/chat/completions` | POST | 聊天补全（OpenAI 兼容） |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 服务器统计信息 |

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t githubcp .
```

### 运行容器

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

## 📈 性能调优

### 推荐配置

| 场景 | Workers | 最大连接数 | 预估并发 |
|------|---------|-----------|---------|
| 开发测试 | 1 | 100 | ~100-200 |
| 小规模部署 | 4 | 200 | ~400-800 |
| 生产环境 | 8+ | 500 | ~1000+ |

### 生产环境部署

```bash
# 使用 gunicorn
gunicorn server:app \
  -w 8 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

> ⚠️ 实际瓶颈在 GitHub Copilot API 本身的速率限制

## 🔒 安全建议

1. **使用 HTTPS**：生产环境建议配合 Nginx/Caddy 使用 HTTPS
2. **限制访问**：通过防火墙或 Nginx 限制访问来源
3. **保护 Token**：不要将 `.github_tokens.json` 文件暴露
4. **定期轮换**：定期清理不使用的 API Key

## ⚠️ 注意事项

1. **需要 Copilot 订阅**：你需要有 GitHub Copilot 订阅才能使用
2. **速率限制**：GitHub Copilot API 有使用限制，请合理使用
3. **合规性**：请确保符合 [GitHub Copilot 使用条款](https://docs.github.com/en/copilot/overview-of-github-copilot/about-github-copilot-individual#github-copilot-acceptable-use-policy)

## 🐛 常见问题

### 授权失败？

- 确保你有有效的 GitHub Copilot 订阅
- 检查网络是否能访问 GitHub

### 模型不可用？

- 不同的 Copilot 订阅可能支持不同的模型
- 尝试使用 `gpt-4o` 作为默认模型

### 请求被限流？

- 调整 `RATE_LIMIT_REQUESTS` 环境变量
- GitHub Copilot API 本身也有速率限制

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[MIT License](LICENSE)

---

<p align="center">
  如果这个项目对你有帮助，请给一个 ⭐️
</p>
