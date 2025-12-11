#!/usr/bin/env python3
"""
GitHub Copilot API to OpenAI API Adapter Server

This server converts GitHub Copilot API to OpenAI-compatible API format,
allowing third-party clients to use GitHub Copilot as an OpenAI backend.

Features:
- OAuth Device Flow for automatic token acquisition
- OpenAI-compatible API endpoints
- Streaming support
"""

import os
import json
import time
import uuid
import asyncio
import secrets
import hashlib
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict
import threading

import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============== Configuration ==============

GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com"
GITHUB_API_BASE = "https://api.github.com"

# GitHub OAuth App - 使用 GitHub Copilot 的 Client ID
# 这是 GitHub CLI 使用的 OAuth App Client ID，支持 Copilot
GITHUB_OAUTH_CLIENT_ID = os.getenv("GITHUB_OAUTH_CLIENT_ID", "Iv1.b507a08c87ecfe98")

# Token 存储路径
TOKEN_STORAGE_PATH = Path(os.getenv("TOKEN_STORAGE_PATH", ".github_tokens.json"))

# 模型缓存时间（秒）
MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "3600"))  # 默认 1 小时

# ============== Performance Configuration ==============

# HTTP 连接池配置
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))  # 最大连接数
HTTP_MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", "20"))  # 保持活跃连接数
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))  # 请求超时（秒）

# 客户端缓存配置
MAX_CLIENTS_CACHE = int(os.getenv("MAX_CLIENTS_CACHE", "1000"))  # 最大缓存客户端数

# 速率限制配置
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # 每分钟请求数
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # 时间窗口（秒）

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Models Manager ==============

# ============== Rate Limiter ==============

class RateLimiter:
    """Simple in-memory rate limiter using sliding window"""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the given key"""
        if not RATE_LIMIT_ENABLED:
            return True
        
        async with self._lock:
            now = time.time()
            if key not in self._requests:
                self._requests[key] = []
            
            # 清理过期的请求记录
            self._requests[key] = [t for t in self._requests[key] if now - t < self.window]
            
            if len(self._requests[key]) >= self.max_requests:
                return False
            
            self._requests[key].append(now)
            return True
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key"""
        now = time.time()
        if key not in self._requests:
            return self.max_requests
        valid_requests = [t for t in self._requests[key] if now - t < self.window]
        return max(0, self.max_requests - len(valid_requests))


rate_limiter = RateLimiter()


# ============== HTTP Client Pool ==============

# 检查是否支持 HTTP/2
try:
    import h2
    HTTP2_AVAILABLE = True
except ImportError:
    HTTP2_AVAILABLE = False
    logger.info("HTTP/2 not available. Install with: pip install httpx[http2]")


def create_http_client(timeout: float = HTTP_TIMEOUT) -> httpx.AsyncClient:
    """Create an HTTP client with connection pool limits"""
    limits = httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE,
    )
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=HTTP2_AVAILABLE,  # 仅在可用时启用 HTTP/2
    )


class CopilotModelsManager:
    """Dynamically fetch and cache available models from GitHub Copilot API"""
    
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._models_list: List[Dict[str, Any]] = []
        self._last_fetch: float = 0
        self._cache_ttl = MODEL_CACHE_TTL
        self._lock = asyncio.Lock()
        self.http_client = create_http_client(30.0)
    
    async def close(self):
        await self.http_client.aclose()
    
    async def get_models(self, github_token: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get available models, fetching from API if cache expired"""
        current_time = time.time()
        
        # Return cached models if still valid (无锁快速路径)
        if not force_refresh and self._models_list and (current_time - self._last_fetch) < self._cache_ttl:
            return self._models_list
        
        # 使用锁防止并发刷新
        async with self._lock:
            # 双重检查
            if not force_refresh and self._models_list and (current_time - self._last_fetch) < self._cache_ttl:
                return self._models_list
            
            # Fetch from API
            try:
                models = await self._fetch_models_from_api(github_token)
                self._models_list = models
                self._models = {m["id"]: m for m in models}
                self._last_fetch = time.time()
                logger.info(f"Models cache refreshed: {len(models)} models")
                return self._models_list
            except Exception as e:
                # If fetch fails and we have cache, return cache
                if self._models_list:
                    logger.warning(f"Failed to refresh models, using cache: {e}")
                    return self._models_list
                raise e
    
    async def get_model(self, model_id: str, github_token: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by ID"""
        await self.get_models(github_token)  # Ensure cache is populated
        return self._models.get(model_id)
    
    async def _fetch_models_from_api(self, github_token: str) -> List[Dict[str, Any]]:
        """Fetch models from GitHub Copilot API"""
        # 首先获取 Copilot token
        copilot_token = await self._get_copilot_token(github_token)
        
        headers = {
            "Authorization": f"Bearer {copilot_token}",
            "Accept": "application/json",
            "User-Agent": "GitHubCopilot/1.0",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.0.0",
        }
        
        # 尝试从 Copilot API 获取模型列表
        response = await self.http_client.get(
            f"{GITHUB_COPILOT_API_BASE}/models",
            headers=headers,
        )
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_models_response(data)
        
        # 如果直接获取失败，尝试从 GitHub Models API 获取
        return await self._fetch_from_github_models(github_token)
    
    async def _get_copilot_token(self, github_token: str) -> str:
        """Get Copilot API token from GitHub token"""
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
            "User-Agent": "GitHubCopilot/1.0",
        }
        
        response = await self.http_client.get(
            f"{GITHUB_API_BASE}/copilot_internal/v2/token",
            headers=headers,
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get Copilot token: {response.text}"
            )
        
        data = response.json()
        return data.get("token")
    
    async def _fetch_from_github_models(self, github_token: str) -> List[Dict[str, Any]]:
        """Fallback: fetch models from GitHub Models API"""
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/json",
            "User-Agent": "GitHubCopilot/1.0",
        }
        
        # GitHub Models API endpoint
        response = await self.http_client.get(
            "https://api.github.com/marketplace/models",
            headers=headers,
        )
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_github_models_response(data)
        
        # 如果所有 API 都失败，返回默认模型列表
        return self._get_fallback_models()
    
    def _parse_models_response(self, data: Any) -> List[Dict[str, Any]]:
        """Parse models response from Copilot API"""
        models = []
        
        # Handle different response formats
        if isinstance(data, dict):
            if "data" in data:
                items = data["data"]
            elif "models" in data:
                items = data["models"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return self._get_fallback_models()
        
        for item in items:
            model = self._normalize_model(item)
            if model:
                models.append(model)
        
        return models if models else self._get_fallback_models()
    
    def _parse_github_models_response(self, data: Any) -> List[Dict[str, Any]]:
        """Parse models response from GitHub Models API"""
        models = []
        
        items = data if isinstance(data, list) else data.get("models", [])
        
        for item in items:
            model_id = item.get("name") or item.get("id") or item.get("model_name")
            if not model_id:
                continue
            
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": item.get("publisher", "github-copilot"),
                "description": item.get("description", f"{model_id} via GitHub Copilot"),
                "context_window": item.get("context_window") or item.get("max_tokens", 128000),
                "supports_vision": item.get("supports_vision", "vision" in model_id.lower() or "4o" in model_id.lower()),
            })
        
        return models if models else self._get_fallback_models()
    
    def _normalize_model(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize model data to OpenAI format"""
        model_id = item.get("id") or item.get("name") or item.get("model")
        if not model_id:
            return None
        
        return {
            "id": model_id,
            "object": "model",
            "created": item.get("created", int(time.time())),
            "owned_by": item.get("owned_by") or item.get("publisher", "github-copilot"),
            "description": item.get("description", f"{model_id} via GitHub Copilot"),
            "context_window": item.get("context_window") or item.get("max_tokens", 128000),
            "supports_vision": item.get("supports_vision", False),
            "capabilities": item.get("capabilities", {}),
        }
    
    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Return fallback models when API fetch fails"""
        fallback = [
            {"id": "gpt-4o", "context_window": 128000, "supports_vision": True},
            {"id": "gpt-4o-mini", "context_window": 128000, "supports_vision": True},
            {"id": "gpt-4", "context_window": 8192, "supports_vision": False},
            {"id": "gpt-3.5-turbo", "context_window": 16385, "supports_vision": False},
            {"id": "claude-3.5-sonnet", "context_window": 200000, "supports_vision": True},
            {"id": "claude-3-opus", "context_window": 200000, "supports_vision": True},
            {"id": "o1-preview", "context_window": 128000, "supports_vision": False},
            {"id": "o1-mini", "context_window": 128000, "supports_vision": False},
        ]
        
        return [
            {
                "id": m["id"],
                "object": "model",
                "created": 1699000000,
                "owned_by": "github-copilot",
                "description": f"{m['id']} via GitHub Copilot",
                "context_window": m["context_window"],
                "supports_vision": m["supports_vision"],
            }
            for m in fallback
        ]
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by ID from cache"""
        return self._models.get(model_id)
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available (in cache)"""
        return model_id in self._models
    
    def get_available_model_ids(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self._models.keys())


# 全局模型管理器
models_manager = CopilotModelsManager()


# ============== Token Storage ==============

class TokenStorage:
    """Persistent storage for GitHub tokens (thread-safe)"""
    
    def __init__(self, path: Path = TOKEN_STORAGE_PATH):
        self.path = path
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load tokens from file"""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self._tokens = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")
                self._tokens = {}
    
    def _save(self):
        """Save tokens to file"""
        try:
            with open(self.path, "w") as f:
                json.dump(self._tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    def get_default_token(self) -> Optional[str]:
        """Get the default (first) token"""
        with self._lock:
            if self._tokens:
                first_key = next(iter(self._tokens))
                return self._tokens[first_key].get("access_token")
        return None
    
    def get_token_by_key(self, key: str) -> Optional[str]:
        """Get token by API key"""
        with self._lock:
            if key in self._tokens:
                return self._tokens[key].get("access_token")
        return None
    
    def save_token(self, access_token: str, user_info: Dict[str, Any] = None) -> str:
        """Save token and return API key"""
        with self._lock:
            # 生成一个简单的 API key
            api_key = f"ghu_{secrets.token_hex(16)}"
            self._tokens[api_key] = {
                "access_token": access_token,
                "user_info": user_info,
                "created_at": time.time(),
            }
            self._save()
            return api_key
    
    def list_tokens(self) -> List[Dict[str, Any]]:
        """List all saved tokens (without exposing actual tokens)"""
        with self._lock:
            result = []
            for key, data in self._tokens.items():
                user_info = data.get("user_info", {})
                result.append({
                    "api_key": key,
                    "username": user_info.get("login", "unknown"),
                    "created_at": data.get("created_at"),
                })
            return result
    
    def delete_token(self, api_key: str) -> bool:
        """Delete a token by API key"""
        with self._lock:
            if api_key in self._tokens:
                del self._tokens[api_key]
                self._save()
                return True
            return False


token_storage = TokenStorage()


# ============== OAuth Device Flow ==============

class OAuthDeviceFlow:
    """Handle GitHub OAuth Device Flow"""
    
    def __init__(self, client_id: str = GITHUB_OAUTH_CLIENT_ID):
        self.client_id = client_id
        self.http_client = create_http_client(30.0)
        # 存储进行中的认证流程
        self._pending_flows: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def close(self):
        await self.http_client.aclose()
    
    async def start_device_flow(self) -> Dict[str, Any]:
        """Start the device authorization flow"""
        response = await self.http_client.post(
            "https://github.com/login/device/code",
            data={
                "client_id": self.client_id,
                "scope": "read:user copilot",
            },
            headers={"Accept": "application/json"},
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to start device flow: {response.text}"
            )
        
        data = response.json()
        
        # 保存流程信息
        device_code = data["device_code"]
        self._pending_flows[device_code] = {
            "device_code": device_code,
            "user_code": data["user_code"],
            "verification_uri": data["verification_uri"],
            "expires_in": data["expires_in"],
            "interval": data.get("interval", 5),
            "started_at": time.time(),
        }
        
        return {
            "device_code": device_code,
            "user_code": data["user_code"],
            "verification_uri": data["verification_uri"],
            "verification_uri_complete": data.get("verification_uri_complete", 
                f"{data['verification_uri']}?user_code={data['user_code']}"),
            "expires_in": data["expires_in"],
            "interval": data.get("interval", 5),
            "message": f"Visit {data['verification_uri']} and enter code: {data['user_code']}"
        }
    
    async def poll_for_token(self, device_code: str) -> Dict[str, Any]:
        """Poll GitHub for the access token"""
        flow_info = self._pending_flows.get(device_code)
        if not flow_info:
            raise HTTPException(status_code=404, detail="Device code not found or expired")
        
        # 检查是否过期
        elapsed = time.time() - flow_info["started_at"]
        if elapsed > flow_info["expires_in"]:
            del self._pending_flows[device_code]
            raise HTTPException(status_code=410, detail="Device code expired")
        
        response = await self.http_client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": self.client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )
        
        data = response.json()
        
        if "error" in data:
            error = data["error"]
            if error == "authorization_pending":
                return {
                    "status": "pending",
                    "message": "Waiting for user authorization...",
                    "interval": flow_info["interval"],
                }
            elif error == "slow_down":
                flow_info["interval"] = data.get("interval", flow_info["interval"] + 5)
                return {
                    "status": "slow_down",
                    "message": "Too many requests, please try again later",
                    "interval": flow_info["interval"],
                }
            elif error == "expired_token":
                del self._pending_flows[device_code]
                raise HTTPException(status_code=410, detail="Device code expired")
            elif error == "access_denied":
                del self._pending_flows[device_code]
                raise HTTPException(status_code=403, detail="User denied access")
            else:
                raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
        
        # 成功获取 token
        access_token = data["access_token"]
        
        # 获取用户信息
        user_info = await self._get_user_info(access_token)
        
        # 保存 token
        api_key = token_storage.save_token(access_token, user_info)
        
        # 清理流程
        del self._pending_flows[device_code]
        
        return {
            "status": "success",
            "api_key": api_key,
            "username": user_info.get("login"),
            "message": f"Authentication successful! User: {user_info.get('login')}",
            "usage": {
                "api_base": "http://localhost:8000/v1",
                "api_key": api_key,
                "note": "Use this api_key as Authorization Bearer token"
            }
        }
    
    async def _get_user_info(self, token: str) -> Dict[str, Any]:
        """Get GitHub user info"""
        response = await self.http_client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        if response.status_code == 200:
            return response.json()
        return {}
    
    async def auto_poll(self, device_code: str, max_wait: int = 300) -> Dict[str, Any]:
        """Automatically poll until success or timeout"""
        flow_info = self._pending_flows.get(device_code)
        if not flow_info:
            raise HTTPException(status_code=404, detail="Device code not found")
        
        start_time = time.time()
        interval = flow_info["interval"]
        
        while time.time() - start_time < max_wait:
            result = await self.poll_for_token(device_code)
            
            if result.get("status") == "success":
                return result
            elif result.get("status") == "slow_down":
                interval = result.get("interval", interval + 5)
            
            await asyncio.sleep(interval)
        
        raise HTTPException(status_code=408, detail="Polling timeout")


oauth_flow = OAuthDeviceFlow()


# ============== Pydantic Models ==============

class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for vision
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Any  # string or list of strings
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


# ============== GitHub Copilot Client ==============

class GitHubCopilotClient:
    """Client for interacting with GitHub Copilot API"""
    
    def __init__(self, token: str):
        self.token = token
        self.copilot_token: Optional[str] = None
        self.copilot_token_expires: float = 0
        self._token_lock = asyncio.Lock()
        self.http_client = create_http_client(HTTP_TIMEOUT)
        self._last_used = time.time()
    
    def touch(self):
        """Update last used time"""
        self._last_used = time.time()
    
    async def close(self):
        await self.http_client.aclose()
    
    async def get_copilot_token(self) -> str:
        """Get or refresh the Copilot API token"""
        current_time = time.time()
        
        # Return cached token if still valid (无锁快速路径)
        if self.copilot_token and current_time < self.copilot_token_expires - 300:
            return self.copilot_token
        
        # 使用锁防止并发刷新
        async with self._token_lock:
            # 双重检查
            if self.copilot_token and current_time < self.copilot_token_expires - 300:
                return self.copilot_token
            
            # Get new token from GitHub
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/json",
                "User-Agent": "GitHubCopilot/1.0",
            }
            
            response = await self.http_client.get(
                f"{GITHUB_API_BASE}/copilot_internal/v2/token",
                headers=headers,
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get Copilot token: {response.text}"
                )
            
            data = response.json()
            self.copilot_token = data.get("token")
            self.copilot_token_expires = data.get("expires_at", current_time + 3600)
            logger.debug(f"Copilot token refreshed, expires at {self.copilot_token_expires}")
            
            return self.copilot_token
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[bytes, None] | Dict[str, Any]:
        """Send chat completion request to Copilot API"""
        token = await self.get_copilot_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "GitHubCopilot/1.0",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.0.0",
            "Openai-Organization": "github-copilot",
            "Openai-Intent": "conversation-panel",
        }
        
        # Build request body
        body = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
        }
        
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.presence_penalty:
            body["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty:
            body["frequency_penalty"] = request.frequency_penalty
        
        if request.stream:
            return self._stream_chat_completion(headers, body)
        else:
            return await self._non_stream_chat_completion(headers, body)
    
    async def _non_stream_chat_completion(
        self,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Non-streaming chat completion"""
        response = await self.http_client.post(
            f"{GITHUB_COPILOT_API_BASE}/chat/completions",
            headers=headers,
            json=body,
        )
        
        if response.status_code != 200:
            error_msg = response.text
            
            # Parse and enhance error message
            try:
                error_json = json.loads(error_msg)
                error_detail = error_json.get("error", {})
                error_code = error_detail.get("code", "unknown_error")
                error_message = error_detail.get("message", error_msg)
                
                # Provide helpful suggestions for common errors
                if error_code in ("model_not_supported", "unsupported_api_for_model"):
                    model_name = body.get("model", "unknown")
                    suggested_models = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "o1-mini"]
                    error_message = (
                        f"Model '{model_name}' is not supported for chat completions. "
                        f"Try using one of these models: {', '.join(suggested_models)}. "
                        f"Use GET /v1/models to see all available models."
                    )
            except json.JSONDecodeError:
                error_message = error_msg
            
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message
            )
        
        return response.json()
    
    async def _stream_chat_completion(
        self,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        """Streaming chat completion"""
        headers["Accept"] = "text/event-stream"
        
        async with self.http_client.stream(
            "POST",
            f"{GITHUB_COPILOT_API_BASE}/chat/completions",
            headers=headers,
            json=body,
        ) as response:
            if response.status_code != 200:
                # Read error before yielding anything
                error_text = await response.aread()
                error_msg = error_text.decode()
                
                # Parse and enhance error message
                try:
                    error_json = json.loads(error_msg)
                    error_detail = error_json.get("error", {})
                    error_code = error_detail.get("code", "unknown_error")
                    error_message = error_detail.get("message", error_msg)
                    
                    # Provide helpful suggestions for common errors
                    if error_code in ("model_not_supported", "unsupported_api_for_model"):
                        model_name = body.get("model", "unknown")
                        suggested_models = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "o1-mini"]
                        error_message = (
                            f"Model '{model_name}' is not supported for chat completions. "
                            f"Try using one of these models: {', '.join(suggested_models)}. "
                            f"Use GET /v1/models to see all available models."
                        )
                except json.JSONDecodeError:
                    error_code = "api_error"
                    error_message = error_msg
                
                # Yield error as SSE event instead of raising exception
                # This allows the client to receive the error gracefully
                error_data = {
                    "error": {
                        "message": error_message,
                        "type": error_code,
                        "code": response.status_code
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n".encode()


# ============== Token Manager ==============

class LRUClientCache:
    """LRU cache for GitHubCopilotClient instances"""
    
    def __init__(self, max_size: int = MAX_CLIENTS_CACHE):
        self.max_size = max_size
        self._cache: OrderedDict[str, GitHubCopilotClient] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, token: str) -> GitHubCopilotClient:
        """Get or create a client for the given token"""
        # 快速路径：无锁检查
        if token in self._cache:
            client = self._cache[token]
            client.touch()
            # 移动到末尾（最近使用）
            self._cache.move_to_end(token)
            return client
        
        async with self._lock:
            # 双重检查
            if token in self._cache:
                client = self._cache[token]
                client.touch()
                self._cache.move_to_end(token)
                return client
            
            # 创建新客户端
            client = GitHubCopilotClient(token)
            self._cache[token] = client
            
            # 如果超过最大大小，清理最旧的客户端
            while len(self._cache) > self.max_size:
                oldest_token, oldest_client = self._cache.popitem(last=False)
                await oldest_client.close()
                logger.info(f"Evicted old client from cache, current size: {len(self._cache)}")
            
            return client
    
    async def close_all(self):
        """Close all clients"""
        async with self._lock:
            for client in self._cache.values():
                await client.close()
            self._cache.clear()


class TokenManager:
    """Manage multiple GitHub tokens and their Copilot clients with LRU cache"""
    
    def __init__(self):
        self._client_cache = LRUClientCache()
    
    async def get_client(self, token: str) -> GitHubCopilotClient:
        """Get or create a client for the given token"""
        return await self._client_cache.get(token)
    
    async def close_all(self):
        """Close all clients"""
        await self._client_cache.close_all()


token_manager = TokenManager()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    yield
    await token_manager.close_all()
    await oauth_flow.close()
    await models_manager.close()


app = FastAPI(
    title="GitHub Copilot OpenAI Adapter",
    description="Converts GitHub Copilot API to OpenAI-compatible format with OAuth support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for third-party clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_token(authorization: Optional[str]) -> str:
    """Extract GitHub token from Authorization header or stored tokens"""
    token = None
    
    if authorization:
        # 提取 key
        if authorization.startswith("Bearer "):
            key = authorization[7:]
        elif authorization.startswith("token "):
            key = authorization[6:]
        else:
            key = authorization
        
        # 首先检查是否是我们生成的 API key (ghu_ 开头)
        if key.startswith("ghu_"):
            token = token_storage.get_token_by_key(key)
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key. Please re-authenticate via /auth/device"
                )
        else:
            # 直接使用 GitHub token
            token = key
    
    if not token:
        # 尝试从存储中获取默认 token
        token = token_storage.get_default_token()
    
    if not token:
        # 尝试环境变量
        token = os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No authentication found. Please authenticate via /auth/device or provide a GitHub token"
        )
    
    return token


# ============== OAuth Endpoints ==============

@app.get("/auth/device", response_class=HTMLResponse)
async def auth_device_page():
    """Show device authorization page with interactive UI and i18n support"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GitHub Copilot OAuth</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #1a1d21;
                --bg-secondary: #23272e;
                --bg-tertiary: #2d333b;
                --bg-hover: #373e47;
                --border-color: #3d444d;
                --border-light: #4a525c;
                --text-primary: #f0f3f6;
                --text-secondary: #b4bbc3;
                --text-muted: #8b929a;
                --accent-primary: #f9a825;
                --accent-secondary: #ffc107;
                --accent-gradient: linear-gradient(135deg, #f9a825 0%, #ff8f00 100%);
                --success-color: #4caf50;
                --success-bg: rgba(76, 175, 80, 0.1);
                --error-color: #f44336;
                --error-bg: rgba(244, 67, 54, 0.1);
                --info-bg: rgba(249, 168, 37, 0.08);
                --shadow-sm: 0 1px 2px rgba(0,0,0,0.2);
                --shadow-md: 0 4px 12px rgba(0,0,0,0.3);
                --shadow-lg: 0 8px 24px rgba(0,0,0,0.4);
                --radius-sm: 6px;
                --radius-md: 10px;
                --radius-lg: 16px;
            }
            
            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body { 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 40px 20px;
                line-height: 1.6;
            }
            
            .container {
                width: 100%;
                max-width: 480px;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 32px;
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .logo-text {
                font-size: 20px;
                font-weight: 700;
                color: var(--text-primary);
            }
            
            .lang-switch {
                display: flex;
                background: var(--bg-secondary);
                border-radius: var(--radius-sm);
                padding: 3px;
                border: 1px solid var(--border-color);
            }
            
            .lang-btn {
                background: transparent;
                border: none;
                color: var(--text-muted);
                padding: 6px 14px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            
            .lang-btn:hover { color: var(--text-secondary); }
            .lang-btn.active { 
                background: var(--bg-tertiary);
                color: var(--text-primary);
                box-shadow: var(--shadow-sm);
            }
            
            .card {
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-lg);
                padding: 32px;
                margin-bottom: 24px;
                box-shadow: var(--shadow-md);
            }
            
            .card-title {
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 12px;
                color: var(--text-primary);
            }
            
            .card-desc {
                color: var(--text-secondary);
                margin-bottom: 24px;
                font-size: 15px;
            }
            
            .btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                width: 100%;
                padding: 14px 24px;
                border-radius: var(--radius-md);
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                border: none;
                font-family: inherit;
            }
            
            .btn-primary {
                background: var(--accent-gradient);
                color: #1a1d21;
                box-shadow: var(--shadow-sm), 0 0 0 0 rgba(249, 168, 37, 0);
            }
            
            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: var(--shadow-md), 0 0 20px rgba(249, 168, 37, 0.3);
            }
            
            .btn-primary:active { transform: translateY(0); }
            
            .btn-secondary {
                background: var(--bg-tertiary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
            }
            
            .btn-secondary:hover {
                background: var(--bg-hover);
                border-color: var(--border-light);
            }
            
            .btn-ghost {
                background: transparent;
                color: var(--text-secondary);
                padding: 10px 16px;
            }
            
            .btn-ghost:hover { color: var(--text-primary); }
            
            .step-list {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .step {
                display: flex;
                align-items: flex-start;
                gap: 16px;
            }
            
            .step-num {
                width: 32px;
                height: 32px;
                background: var(--accent-gradient);
                color: #1a1d21;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 14px;
                flex-shrink: 0;
                box-shadow: var(--shadow-sm);
            }
            
            .step-content {
                flex: 1;
                padding-top: 4px;
            }
            
            .step-title {
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 8px;
            }
            
            .code-display {
                background: var(--bg-primary);
                border: 2px dashed var(--border-light);
                border-radius: var(--radius-md);
                padding: 24px;
                text-align: center;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .code-display:hover {
                border-color: var(--accent-primary);
                background: var(--info-bg);
            }
            
            .code-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 36px;
                font-weight: 700;
                letter-spacing: 8px;
                color: var(--accent-primary);
                text-shadow: 0 0 30px rgba(249, 168, 37, 0.3);
            }
            
            .code-hint {
                font-size: 12px;
                color: var(--text-muted);
                margin-top: 8px;
            }
            
            .status-box {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                padding: 16px;
                border-radius: var(--radius-md);
                margin: 8px 0;
                font-weight: 500;
            }
            
            .status-box.pending {
                background: var(--info-bg);
                border: 1px solid rgba(249, 168, 37, 0.2);
                color: var(--accent-secondary);
            }
            
            .status-box.success {
                background: var(--success-bg);
                border: 1px solid rgba(76, 175, 80, 0.2);
                color: var(--success-color);
            }
            
            .status-box.error {
                background: var(--error-bg);
                border: 1px solid rgba(244, 67, 54, 0.2);
                color: var(--error-color);
            }
            
            .spinner {
                width: 18px;
                height: 18px;
                border: 2px solid rgba(249, 168, 37, 0.2);
                border-top-color: var(--accent-primary);
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }
            
            @keyframes spin { to { transform: rotate(360deg); } }
            
            .success-icon {
                width: 64px;
                height: 64px;
                background: var(--success-bg);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                margin: 0 auto 24px;
                border: 2px solid rgba(76, 175, 80, 0.3);
            }
            
            .api-key-box {
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-md);
                padding: 16px;
                margin: 16px 0;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .api-key-box:hover {
                border-color: var(--accent-primary);
            }
            
            .api-key-label {
                font-size: 12px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }
            
            .api-key-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                color: var(--text-primary);
                word-break: break-all;
                line-height: 1.5;
            }
            
            .usage-section {
                margin-top: 28px;
                padding-top: 28px;
                border-top: 1px solid var(--border-color);
            }
            
            .usage-title {
                font-size: 14px;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 16px;
            }
            
            .usage-grid {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            
            .usage-item {
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-sm);
                padding: 12px 16px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .usage-item:hover {
                border-color: var(--border-light);
                background: var(--bg-tertiary);
            }
            
            .usage-item-label {
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }
            
            .usage-item-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: var(--text-primary);
            }
            
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 24px;
                border-top: 1px solid var(--border-color);
                color: var(--text-muted);
                font-size: 13px;
            }
            
            .footer a {
                color: var(--accent-primary);
                text-decoration: none;
                transition: color 0.2s;
            }
            
            .footer a:hover { color: var(--accent-secondary); }
            
            .toast {
                position: fixed;
                top: 24px;
                right: 24px;
                background: var(--bg-secondary);
                border: 1px solid var(--success-color);
                color: var(--success-color);
                padding: 12px 20px;
                border-radius: var(--radius-md);
                font-weight: 500;
                font-size: 14px;
                box-shadow: var(--shadow-lg);
                animation: slideIn 0.3s ease;
                z-index: 1000;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(20px); }
                to { opacity: 1; transform: translateX(0); }
            }
            
            .divider {
                height: 1px;
                background: var(--border-color);
                margin: 24px 0;
            }
            
            @media (max-width: 520px) {
                body { padding: 24px 16px; }
                .card { padding: 24px 20px; }
                .code-value { font-size: 28px; letter-spacing: 6px; }
                .header { flex-direction: column; gap: 16px; align-items: stretch; }
                .logo { justify-content: center; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <span class="logo-text" data-i18n="title">Copilot Auth</span>
                </div>
                <div class="lang-switch">
                    <button class="lang-btn active" onclick="setLang('en')" id="lang-en">EN</button>
                    <button class="lang-btn" onclick="setLang('zh')" id="lang-zh">中文</button>
                </div>
            </div>
            
            <div id="init-card" class="card">
                <h2 class="card-title" data-i18n="welcome_title">Connect to GitHub</h2>
                <p class="card-desc" data-i18n="init_desc">Authorize with your GitHub account to get an API key for third-party clients.</p>
                <button class="btn btn-primary" onclick="startAuth()" data-i18n="start_btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                    <span>Start Authorization</span>
                </button>
            </div>
            
            <div id="auth-card" class="card" style="display: none;">
                <div class="step-list">
                    <div class="step">
                        <div class="step-num">1</div>
                        <div class="step-content">
                            <div class="step-title" data-i18n="step1">Open GitHub to authorize</div>
                            <a id="auth-link" href="" target="_blank" class="btn btn-secondary">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                                <span data-i18n="open_github">Open GitHub</span>
                            </a>
                        </div>
                    </div>
                    
                    <div class="step">
                        <div class="step-num">2</div>
                        <div class="step-content">
                            <div class="step-title" data-i18n="step2">Enter verification code</div>
                            <div class="code-display" onclick="copyCode()">
                                <div class="code-value" id="user-code">----</div>
                                <div class="code-hint" data-i18n="click_copy">Click to copy</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="step">
                        <div class="step-num">3</div>
                        <div class="step-content">
                            <div class="step-title" data-i18n="step3">Wait for confirmation</div>
                            <div id="status" class="status-box pending">
                                <div class="spinner"></div>
                                <span data-i18n="waiting">Waiting for authorization...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="divider"></div>
                <button class="btn btn-ghost" onclick="location.reload()" data-i18n="restart">← Start Over</button>
            </div>
            
            <div id="success-card" class="card" style="display: none;">
                <div class="success-icon">✓</div>
                <h2 class="card-title" style="text-align: center;" data-i18n="success_title">Authorization Successful</h2>
                <p class="card-desc" style="text-align: center;">
                    <span data-i18n="welcome_user">Welcome,</span> <strong id="username"></strong>
                </p>
                
                <div class="api-key-box" onclick="copyApiKey()">
                    <div class="api-key-label" data-i18n="your_api_key">Your API Key</div>
                    <div class="api-key-value" id="api-key"></div>
                </div>
                
                <button class="btn btn-primary" onclick="copyApiKey()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
                    <span data-i18n="copy_key">Copy API Key</span>
                </button>
                
                <div class="usage-section">
                    <div class="usage-title" data-i18n="usage_title">Configuration</div>
                    <div class="usage-grid">
                        <div class="usage-item" onclick="copyText('http://localhost:8000/v1')">
                            <div class="usage-item-label">API Base URL</div>
                            <div class="usage-item-value">http://localhost:8000/v1</div>
                        </div>
                        <div class="usage-item" onclick="copyApiKey()">
                            <div class="usage-item-label">API Key</div>
                            <div class="usage-item-value" id="usage-api-key"></div>
                        </div>
                    </div>
                </div>
                
                <div class="divider"></div>
                <button class="btn btn-ghost" onclick="location.reload()" data-i18n="auth_another">+ Add Another Account</button>
            </div>
            
            <div class="footer">
                <p>GitHub Copilot OpenAI Adapter</p>
            </div>
        </div>
        
        <script>
            const i18n = {
                en: {
                    title: 'Copilot Auth',
                    welcome_title: 'Connect to GitHub',
                    init_desc: 'Authorize with your GitHub account to get an API key for third-party clients.',
                    start_btn: 'Start Authorization',
                    step1: 'Open GitHub to authorize',
                    step2: 'Enter verification code',
                    step3: 'Wait for confirmation',
                    open_github: 'Open GitHub',
                    click_copy: 'Click to copy',
                    waiting: 'Waiting for authorization...',
                    restart: '← Start Over',
                    success_title: 'Authorization Successful',
                    welcome_user: 'Welcome,',
                    your_api_key: 'Your API Key',
                    copy_key: 'Copy API Key',
                    usage_title: 'Configuration',
                    auth_another: '+ Add Another Account',
                    copied: 'Copied to clipboard!',
                    auth_failed: 'Authorization failed: ',
                    pending: 'Waiting for authorization...',
                    slow_down: 'Rate limited, retrying...'
                },
                zh: {
                    title: 'Copilot 授权',
                    welcome_title: '连接到 GitHub',
                    init_desc: '使用 GitHub 账号授权，获取 API Key 供第三方客户端使用。',
                    start_btn: '开始授权',
                    step1: '打开 GitHub 进行授权',
                    step2: '输入验证码',
                    step3: '等待确认',
                    open_github: '打开 GitHub',
                    click_copy: '点击复制',
                    waiting: '等待授权中...',
                    restart: '← 重新开始',
                    success_title: '授权成功',
                    welcome_user: '欢迎，',
                    your_api_key: '你的 API Key',
                    copy_key: '复制 API Key',
                    usage_title: '配置信息',
                    auth_another: '+ 添加其他账号',
                    copied: '已复制到剪贴板！',
                    auth_failed: '授权失败：',
                    pending: '等待用户授权...',
                    slow_down: '请求频繁，正在重试...'
                }
            };
            
            let currentLang = localStorage.getItem('lang') || 'en';
            let deviceCode = null;
            let pollInterval = 5;
            
            function setLang(lang) {
                currentLang = lang;
                localStorage.setItem('lang', lang);
                document.getElementById('lang-en').classList.toggle('active', lang === 'en');
                document.getElementById('lang-zh').classList.toggle('active', lang === 'zh');
                document.querySelectorAll('[data-i18n]').forEach(el => {
                    const key = el.getAttribute('data-i18n');
                    if (i18n[lang][key]) el.textContent = i18n[lang][key];
                });
            }
            
            function t(key) {
                return i18n[currentLang][key] || i18n['en'][key] || key;
            }
            
            function showToast(message) {
                const existing = document.querySelector('.toast');
                if (existing) existing.remove();
                const toast = document.createElement('div');
                toast.className = 'toast';
                toast.textContent = message;
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 2500);
            }
            
            async function startAuth() {
                try {
                    const resp = await fetch('/auth/device/start', { method: 'POST' });
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.detail || 'Failed to start');
                    
                    deviceCode = data.device_code;
                    pollInterval = data.interval;
                    
                    document.getElementById('user-code').textContent = data.user_code;
                    document.getElementById('auth-link').href = data.verification_uri_complete;
                    document.getElementById('init-card').style.display = 'none';
                    document.getElementById('auth-card').style.display = 'block';
                    
                    window.open(data.verification_uri_complete, '_blank');
                    pollForToken();
                } catch (e) {
                    alert(t('auth_failed') + e.message);
                }
            }
            
            async function pollForToken() {
                try {
                    const resp = await fetch(`/auth/device/poll?device_code=${deviceCode}`);
                    const data = await resp.json();
                    
                    if (data.status === 'success') {
                        document.getElementById('auth-card').style.display = 'none';
                        document.getElementById('success-card').style.display = 'block';
                        document.getElementById('username').textContent = data.username;
                        document.getElementById('api-key').textContent = data.api_key;
                        document.getElementById('usage-api-key').textContent = data.api_key;
                    } else if (data.status === 'pending' || data.status === 'slow_down') {
                        pollInterval = data.interval || pollInterval;
                        const statusEl = document.getElementById('status');
                        statusEl.innerHTML = '<div class="spinner"></div><span>' + 
                            (data.status === 'slow_down' ? t('slow_down') : t('pending')) + '</span>';
                        setTimeout(pollForToken, pollInterval * 1000);
                    }
                } catch (e) {
                    const statusEl = document.getElementById('status');
                    statusEl.className = 'status-box error';
                    statusEl.innerHTML = '<span>' + t('auth_failed') + e.message + '</span>';
                }
            }
            
            function copyCode() {
                copyText(document.getElementById('user-code').textContent);
            }
            
            function copyApiKey() {
                copyText(document.getElementById('api-key').textContent);
            }
            
            function copyText(text) {
                navigator.clipboard.writeText(text).then(() => showToast(t('copied')));
            }
            
            document.addEventListener('DOMContentLoaded', () => setLang(currentLang));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/auth/device/start")
async def auth_device_start():
    """Start the OAuth device flow"""
    return await oauth_flow.start_device_flow()


@app.get("/auth/device/poll")
async def auth_device_poll(device_code: str = Query(...)):
    """Poll for OAuth token"""
    return await oauth_flow.poll_for_token(device_code)


@app.post("/auth/device/wait")
async def auth_device_wait(device_code: str = Query(...), timeout: int = Query(300)):
    """Wait for OAuth completion (blocking)"""
    return await oauth_flow.auto_poll(device_code, timeout)


@app.get("/auth/tokens")
async def list_tokens():
    """List all saved tokens"""
    return {"tokens": token_storage.list_tokens()}


@app.delete("/auth/tokens/{api_key}")
async def delete_token(api_key: str):
    """Delete a saved token"""
    if token_storage.delete_token(api_key):
        return {"message": "Token deleted"}
    raise HTTPException(status_code=404, detail="Token not found")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GitHub Copilot OpenAI Adapter",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth/device (OAuth Device Authorization)",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
        },
        "quick_start": {
            "step1": "Visit /auth/device for GitHub OAuth authorization",
            "step2": "Get your API Key",
            "step3": "Use API Key as Bearer token to call /v1/chat/completions"
        }
    }


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None),
    refresh: bool = Query(False, description="Force refresh models from API"),
):
    """List available models (OpenAI compatible) - dynamically fetched from GitHub Copilot API"""
    try:
        token = extract_token(authorization)
        models = await models_manager.get_models(token, force_refresh=refresh)
        return {
            "object": "list",
            "data": models
        }
    except HTTPException:
        # If no auth, return fallback models
        return {
            "object": "list",
            "data": models_manager._get_fallback_models()
        }


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    authorization: Optional[str] = Header(None),
):
    """Get specific model info (OpenAI compatible)"""
    try:
        token = extract_token(authorization)
        model = await models_manager.get_model(model_id, token)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return model
    except HTTPException as e:
        if e.status_code == 401:
            # Check fallback models
            fallback = {m["id"]: m for m in models_manager._get_fallback_models()}
            if model_id in fallback:
                return fallback[model_id]
        raise


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Chat completions endpoint (OpenAI compatible)"""
    token = extract_token(authorization)
    
    # 速率限制检查
    rate_key = hashlib.md5(token.encode()).hexdigest()[:16]
    if not await rate_limiter.is_allowed(rate_key):
        remaining = rate_limiter.get_remaining(rate_key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {RATE_LIMIT_WINDOW} seconds.",
            headers={
                "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(time.time()) + RATE_LIMIT_WINDOW),
            }
        )
    
    # 检查模型是否在已知列表中（仅记录警告，不阻止请求）
    if models_manager._models and not models_manager.is_model_available(request.model):
        available_models = models_manager.get_available_model_ids()
        logger.warning(
            f"Model '{request.model}' not found in cached models. "
            f"Available models: {available_models}. "
            f"Request will proceed but may fail."
        )
    
    client = await token_manager.get_client(token)
    
    try:
        if request.stream:
            generator = await client.chat_completion(request)
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-RateLimit-Remaining": str(rate_limiter.get_remaining(rate_key)),
                }
            )
        else:
            result = await client.chat_completion(request)
            return JSONResponse(
                content=result,
                headers={
                    "X-RateLimit-Remaining": str(rate_limiter.get_remaining(rate_key)),
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None),
):
    """Embeddings endpoint (limited support - may not be available in Copilot)"""
    raise HTTPException(
        status_code=501,
        detail="Embeddings are not supported by GitHub Copilot API"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "config": {
            "max_connections": HTTP_MAX_CONNECTIONS,
            "max_clients_cache": MAX_CLIENTS_CACHE,
            "rate_limit_enabled": RATE_LIMIT_ENABLED,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window": RATE_LIMIT_WINDOW,
            "model_cache_ttl": MODEL_CACHE_TTL,
        },
        "cache": {
            "models_cached": len(models_manager._models_list),
            "models_last_fetch": models_manager._last_fetch,
        }
    }


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))  # 默认单进程
    
    # 推荐 worker 数量
    cpu_count = multiprocessing.cpu_count()
    recommended_workers = min(cpu_count * 2 + 1, 8)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║         GitHub Copilot → OpenAI API Adapter Server               ║
╠════════════════════════════════════════════════════════════════════╣
║  Server:    http://{host}:{port}                                      
║  API Docs:  http://{host}:{port}/docs                                 
║  OAuth:     http://{host}:{port}/auth/device                          
║  Stats:     http://{host}:{port}/stats                                
╠════════════════════════════════════════════════════════════════════╣
║  ⚡ 性能配置:                                                       ║
║  - Workers: {workers} (推荐: {recommended_workers}, CPU核心数: {cpu_count})              
║  - 最大连接数: {HTTP_MAX_CONNECTIONS}                                        
║  - 客户端缓存: {MAX_CLIENTS_CACHE}                                         
║  - 速率限制: {RATE_LIMIT_REQUESTS} 请求/{RATE_LIMIT_WINDOW}秒                       
╠════════════════════════════════════════════════════════════════════╣
║  📈 生产环境部署建议:                                             ║
║  gunicorn server:app -w {recommended_workers} -k uvicorn.workers.UvicornWorker ║
╠════════════════════════════════════════════════════════════════════╣
║  🔐 快速开始:                                                     ║
║  1. 打开浏览器访问 http://localhost:{port}/auth/device              
║  2. 点击授权按钮，完成 GitHub OAuth 登录                          ║
║  3. 获取 API Key (ghu_xxx 格式)                                   ║
║  4. 在第三方客户端配置:                                           ║
║     - API Base: http://localhost:{port}/v1                         
║     - API Key:  你获取的 ghu_xxx                                  ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    if workers > 1:
        # 多进程模式
        uvicorn.run("server:app", host=host, port=port, workers=workers)
    else:
        uvicorn.run(app, host=host, port=port)
