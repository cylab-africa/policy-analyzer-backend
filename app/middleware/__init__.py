"""
Middleware package initialization
"""
from .auth import APIKeyMiddleware

__all__ = ["APIKeyMiddleware"]