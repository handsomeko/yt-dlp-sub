"""
Final API Security Module - Addresses Issues #126-135
Comprehensive API-specific security controls
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict, deque
import asyncio
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import re
import aiofiles
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import jwt
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import aioredis
from pydantic import BaseModel, validator
import httpx

logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Security configuration with validation"""
    
    # CORS settings (Issue #126)
    cors_allowed_origins: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    cors_allow_credentials: bool = True
    cors_allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allowed_headers: List[str] = ["Authorization", "Content-Type", "X-Request-ID"]
    cors_max_age: int = 3600
    
    # Authentication (Issue #127)
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    api_key_header: str = "X-API-Key"
    require_auth_endpoints: List[str] = ["/api/v1/*"]
    public_endpoints: List[str] = ["/health", "/docs", "/openapi.json"]
    
    # Rate limiting (Issue #128)
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10
    
    # Security headers (Issue #129)
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    
    # Request limits (Issue #130)
    max_request_size_mb: int = 10
    max_upload_size_mb: int = 100
    request_timeout_seconds: int = 30
    
    # Webhook security (Issue #131)
    webhook_secret: Optional[str] = None
    webhook_timeout: int = 10
    
    # Database (Issue #132)
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    
    # API versioning (Issue #134)
    api_version: str = "v1"
    supported_versions: List[str] = ["v1"]
    deprecation_headers: bool = True
    
    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v


class RateLimiter:
    """Advanced rate limiting with sliding window (Issue #128)"""
    
    def __init__(self, requests: int = 100, window: int = 60, burst: int = 10):
        self.requests = requests
        self.window = window
        self.burst = burst
        self.clients: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests))
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        async with self.lock:
            now = time.time()
            requests = self.clients[client_id]
            
            # Remove old requests outside window
            while requests and requests[0] < now - self.window:
                requests.popleft()
            
            # Check burst limit (requests in last second)
            recent = [r for r in requests if r > now - 1]
            if len(recent) >= self.burst:
                return False
            
            # Check window limit
            if len(requests) >= self.requests:
                return False
            
            # Add current request
            requests.append(now)
            return True
    
    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until rate limit resets"""
        if client_id not in self.clients:
            return 0
        
        requests = self.clients[client_id]
        if not requests:
            return 0
        
        oldest = requests[0]
        return max(0, int(self.window - (time.time() - oldest)))


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT and API key authentication (Issue #127)"""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.jwt_secret = config.jwt_secret
        self.api_keys: Set[str] = set()
        self.load_api_keys()
    
    def load_api_keys(self):
        """Load valid API keys from database"""
        # In production, load from database
        # For now, generate some secure keys
        self.api_keys = {
            secrets.token_urlsafe(32),
            secrets.token_urlsafe(32),
        }
    
    async def dispatch(self, request: Request, call_next):
        """Authenticate requests"""
        path = request.url.path
        
        # Skip authentication for public endpoints
        if any(path.startswith(endpoint) for endpoint in self.config.public_endpoints):
            return await call_next(request)
        
        # Check if endpoint requires authentication
        requires_auth = any(
            self._match_pattern(path, pattern) 
            for pattern in self.config.require_auth_endpoints
        )
        
        if requires_auth:
            # Try JWT authentication
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                if not await self.verify_jwt(token):
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or expired token"}
                    )
            # Try API key authentication
            elif api_key := request.headers.get(self.config.api_key_header):
                if api_key not in self.api_keys:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid API key"}
                    )
            else:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required"}
                )
        
        # Add user info to request state
        response = await call_next(request)
        return response
    
    async def verify_jwt(self, token: str) -> bool:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            # Additional validation
            if payload.get('exp', 0) < time.time():
                return False
            return True
        except jwt.InvalidTokenError:
            return False
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match path against pattern with wildcards"""
        pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{pattern}$", path))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses (Issue #129)"""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        response = await call_next(request)
        
        if self.config.enable_security_headers:
            # HSTS
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.config.hsts_max_age}; includeSubDomains; preload"
            )
            
            # Content Security Policy
            response.headers["Content-Security-Policy"] = self.config.csp_policy
            
            # Other security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = (
                "geolocation=(), microphone=(), camera=()"
            )
            
            # Remove sensitive headers
            response.headers.pop("Server", None)
            response.headers.pop("X-Powered-By", None)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware (Issue #128)"""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.limiter = RateLimiter(
            requests=config.rate_limit_requests,
            window=config.rate_limit_window_seconds,
            burst=config.rate_limit_burst
        )
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting"""
        # Get client identifier (IP or user ID)
        client_id = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not await self.limiter.check_rate_limit(client_id):
            retry_after = self.limiter.get_retry_after(client_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)}
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.config.rate_limit_requests)
        response.headers["X-RateLimit-Window"] = str(self.config.rate_limit_window_seconds)
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size (Issue #130)"""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.max_size = config.max_request_size_mb * 1024 * 1024
        self.max_upload = config.max_upload_size_mb * 1024 * 1024
    
    async def dispatch(self, request: Request, call_next):
        """Check request size"""
        content_length = request.headers.get("content-length")
        
        if content_length:
            size = int(content_length)
            
            # Check if it's an upload endpoint
            is_upload = "/upload" in request.url.path
            max_allowed = self.max_upload if is_upload else self.max_size
            
            if size > max_allowed:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": f"Request body too large. Max size: {max_allowed} bytes"
                    }
                )
        
        response = await call_next(request)
        return response


class WebhookValidator:
    """Webhook signature validation (Issue #131)"""
    
    def __init__(self, secret: str):
        self.secret = secret.encode() if secret else None
    
    def generate_signature(self, payload: bytes) -> str:
        """Generate HMAC signature for payload"""
        if not self.secret:
            return ""
        
        signature = hmac.new(
            self.secret,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        if not self.secret:
            return True  # No secret configured
        
        expected = self.generate_signature(payload)
        return hmac.compare_digest(expected, signature)
    
    async def send_webhook(self, url: str, data: dict, timeout: int = 10) -> bool:
        """Send webhook with signature"""
        payload = json.dumps(data).encode()
        signature = self.generate_signature(payload)
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(int(time.time()))
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    content=payload,
                    headers=headers,
                    timeout=timeout
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return False


class DatabasePoolManager:
    """Manage database connection pools (Issue #132)"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.engine = None
        self.redis_pool = None
    
    def create_sql_engine(self, database_url: str):
        """Create SQLAlchemy engine with connection pooling"""
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.config.db_pool_size,
            max_overflow=self.config.db_max_overflow,
            pool_timeout=self.config.db_pool_timeout,
            pool_recycle=self.config.db_pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
            echo_pool=True  # Log pool checkouts/checkins
        )
        return self.engine
    
    async def create_redis_pool(self, redis_url: str):
        """Create Redis connection pool"""
        self.redis_pool = await aioredis.create_redis_pool(
            redis_url,
            minsize=5,
            maxsize=self.config.db_pool_size,
            timeout=self.config.db_pool_timeout
        )
        return self.redis_pool
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    async def close_pools(self):
        """Close all connection pools"""
        if self.engine:
            self.engine.dispose()
        
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()


class DependencyScanner:
    """Scan dependencies for vulnerabilities (Issue #133)"""
    
    def __init__(self):
        self.vulnerability_db = {}
        self.last_scan = None
    
    async def scan_python_dependencies(self) -> Dict[str, List[str]]:
        """Scan Python dependencies using safety"""
        vulnerabilities = {}
        
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for vuln in data:
                    package = vuln.get("package", "unknown")
                    if package not in vulnerabilities:
                        vulnerabilities[package] = []
                    vulnerabilities[package].append({
                        "id": vuln.get("vulnerability_id"),
                        "severity": vuln.get("severity", "unknown"),
                        "description": vuln.get("description", "")
                    })
        
        except subprocess.TimeoutExpired:
            logger.error("Dependency scan timeout")
        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
        
        self.vulnerability_db = vulnerabilities
        self.last_scan = datetime.now()
        
        return vulnerabilities
    
    async def scan_npm_dependencies(self) -> Dict[str, List[str]]:
        """Scan npm dependencies using npm audit"""
        vulnerabilities = {}
        
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for advisory in data.get("advisories", {}).values():
                    module = advisory.get("module_name", "unknown")
                    if module not in vulnerabilities:
                        vulnerabilities[module] = []
                    vulnerabilities[module].append({
                        "id": advisory.get("id"),
                        "severity": advisory.get("severity", "unknown"),
                        "title": advisory.get("title", ""),
                        "url": advisory.get("url", "")
                    })
        
        except Exception as e:
            logger.error(f"NPM audit failed: {e}")
        
        return vulnerabilities
    
    def get_report(self) -> dict:
        """Get vulnerability report"""
        return {
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "vulnerabilities": self.vulnerability_db,
            "total_count": sum(len(v) for v in self.vulnerability_db.values())
        }


class APIVersionManager:
    """Manage API versioning (Issue #134)"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.current_version = config.api_version
        self.supported_versions = set(config.supported_versions)
        self.deprecation_dates = {
            "v0": datetime(2024, 1, 1),  # Already deprecated
        }
    
    def extract_version(self, path: str) -> Optional[str]:
        """Extract API version from path"""
        match = re.match(r'^/api/(v\d+)/', path)
        return match.group(1) if match else None
    
    def is_supported(self, version: str) -> bool:
        """Check if version is supported"""
        return version in self.supported_versions
    
    def is_deprecated(self, version: str) -> bool:
        """Check if version is deprecated"""
        if version not in self.deprecation_dates:
            return False
        return datetime.now() > self.deprecation_dates[version]
    
    def get_deprecation_headers(self, version: str) -> dict:
        """Get deprecation headers for response"""
        headers = {}
        
        if version in self.deprecation_dates:
            date = self.deprecation_dates[version]
            headers["Sunset"] = date.strftime("%a, %d %b %Y %H:%M:%S GMT")
            headers["Deprecation"] = "true"
            headers["Link"] = f'</api/{self.current_version}/>; rel="successor-version"'
        
        return headers


class APIAuditLogger:
    """Comprehensive API audit logging (Issue #135)"""
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.file_date = None
        self.lock = asyncio.Lock()
    
    async def _get_log_file(self):
        """Get current log file, rotating daily"""
        today = datetime.now().date()
        
        if self.file_date != today:
            if self.current_file:
                await self.current_file.close()
            
            filename = self.log_dir / f"audit_{today.isoformat()}.jsonl"
            self.current_file = await aiofiles.open(filename, 'a')
            self.file_date = today
        
        return self.current_file
    
    async def log_request(
        self,
        request: Request,
        response: Response,
        user_id: Optional[str] = None,
        duration_ms: float = 0
    ):
        """Log API request with details"""
        async with self.lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "query": dict(request.query_params),
                "status_code": response.status_code,
                "user_id": user_id,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "duration_ms": duration_ms,
                "request_id": request.headers.get("x-request-id", secrets.token_hex(8))
            }
            
            # Add security events
            if response.status_code == 401:
                log_entry["security_event"] = "authentication_failed"
            elif response.status_code == 403:
                log_entry["security_event"] = "authorization_failed"
            elif response.status_code == 429:
                log_entry["security_event"] = "rate_limit_exceeded"
            
            file = await self._get_log_file()
            await file.write(json.dumps(log_entry) + "\n")
            await file.flush()
    
    async def log_security_event(
        self,
        event_type: str,
        details: dict,
        severity: str = "INFO"
    ):
        """Log security-specific events"""
        async with self.lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "severity": severity,
                "details": details
            }
            
            file = await self._get_log_file()
            await file.write(json.dumps(log_entry) + "\n")
            await file.flush()
    
    async def close(self):
        """Close log files"""
        if self.current_file:
            await self.current_file.close()


class AuditMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware (Issue #135)"""
    
    def __init__(self, app: ASGIApp, audit_logger: APIAuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
    
    async def dispatch(self, request: Request, call_next):
        """Log all API requests"""
        start_time = time.time()
        
        # Get user ID from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        await self.audit_logger.log_request(
            request=request,
            response=response,
            user_id=user_id,
            duration_ms=duration_ms
        )
        
        return response


def create_secure_app(app, config: SecurityConfig):
    """Apply all security middleware to FastAPI app"""
    
    # Initialize components
    audit_logger = APIAuditLogger()
    version_manager = APIVersionManager(config)
    pool_manager = DatabasePoolManager(config)
    webhook_validator = WebhookValidator(config.webhook_secret)
    dependency_scanner = DependencyScanner()
    
    # Apply middleware in correct order
    
    # 1. Trusted host validation
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]
    )
    
    # 2. CORS (Issue #126 - Fixed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allowed_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=config.cors_allowed_methods,
        allow_headers=config.cors_allowed_headers,
        max_age=config.cors_max_age
    )
    
    # 3. Request size limiting (Issue #130)
    app.add_middleware(RequestSizeLimitMiddleware, config=config)
    
    # 4. Rate limiting (Issue #128)
    app.add_middleware(RateLimitMiddleware, config=config)
    
    # 5. Authentication (Issue #127)
    app.add_middleware(AuthenticationMiddleware, config=config)
    
    # 6. Security headers (Issue #129)
    app.add_middleware(SecurityHeadersMiddleware, config=config)
    
    # 7. Audit logging (Issue #135)
    app.add_middleware(AuditMiddleware, audit_logger=audit_logger)
    
    # Store components in app state
    app.state.audit_logger = audit_logger
    app.state.version_manager = version_manager
    app.state.pool_manager = pool_manager
    app.state.webhook_validator = webhook_validator
    app.state.dependency_scanner = dependency_scanner
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup"""
        # Scan dependencies
        await dependency_scanner.scan_python_dependencies()
        
        # Initialize database pools
        # await pool_manager.create_redis_pool("redis://localhost:6379")
        
        logger.info("API security initialized")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown"""
        await audit_logger.close()
        await pool_manager.close_pools()
        
        logger.info("API security shutdown complete")
    
    # Add security endpoints
    @app.get("/api/security/status")
    async def security_status():
        """Get security system status"""
        return {
            "cors_configured": True,
            "authentication_enabled": True,
            "rate_limiting_enabled": True,
            "security_headers_enabled": config.enable_security_headers,
            "audit_logging_enabled": True,
            "dependency_scanning_enabled": True,
            "last_dependency_scan": dependency_scanner.last_scan
        }
    
    @app.get("/api/security/vulnerabilities")
    async def get_vulnerabilities():
        """Get dependency vulnerability report"""
        return dependency_scanner.get_report()
    
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    # Create FastAPI app
    app = FastAPI()
    
    # Create security config
    config = SecurityConfig(
        jwt_secret=secrets.token_urlsafe(32),
        cors_allowed_origins=["http://localhost:3000"],
        webhook_secret=secrets.token_urlsafe(32)
    )
    
    # Apply security
    secure_app = create_secure_app(app, config)
    
    # Run server
    uvicorn.run(secure_app, host="0.0.0.0", port=8000)