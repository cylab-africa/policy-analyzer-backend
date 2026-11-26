"""
Authentication middleware for API key validation
"""
import hmac

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings
from typing import List


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key for protected endpoints.
    Excludes public endpoints like root, docs, and OpenAPI spec.
    """
    
    def __init__(self, app, excluded_paths: List[str] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/docs",
            "/api/docs", 
            "/openapi.json",
            "/redoc"
        ]
    
    async def dispatch(self, request: Request, call_next):
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        api_key = self._extract_api_key(request)
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "API key required",
                    "message": "Please provide a valid API key in the X-API-Key header"
                }
            )
        
        if not self._validate_api_key(api_key):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        return await call_next(request)
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if the request path should be excluded from API key validation"""
        if path in self.excluded_paths:
            return True
        normalized_path = path.rstrip('/')
        for excluded_path in self.excluded_paths:
            if normalized_path == excluded_path.rstrip('/'):
                return True
        
        return False
    
    def _extract_api_key(self, request: Request) -> str:
        """Extract API key from request headers"""
        api_key_headers = [
            "X-API-Key",
            "X-Api-Key", 
            "Authorization",
            "API-Key"
        ]
        
        for header in api_key_headers:
            api_key = request.headers.get(header)
            if api_key:
                if header == "Authorization" and api_key.startswith(("Bearer ", "API-Key ")):
                    api_key = api_key.split(" ", 1)[1]
                return api_key
        
        return None
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate the provided API key against the configured key"""
        expected_api_key = settings.api_key
        
        if not expected_api_key:
            # If no API key is configured, reject all requests
            return False

        # Constant-time comparison so response timing cannot be used to guess the key
        return hmac.compare_digest(api_key.encode("utf-8"), expected_api_key.encode("utf-8"))