from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from .config import DISABLE_CSP

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers that allow web UI access
    while maintaining reasonable security.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add minimal security headers
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Check if CSP should be disabled (for maximum web UI compatibility)
        if not DISABLE_CSP:
            # Only add CSP if explicitly enabled
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Ultra-permissive CSP
            csp_policy_ultra = "default-src *; script-src * 'unsafe-inline' 'unsafe-eval'; style-src * 'unsafe-inline'; img-src * data:; connect-src *; font-src *; object-src *; media-src *; child-src *; frame-src *; worker-src *; form-action *; frame-ancestors *"
            response.headers["Content-Security-Policy"] = csp_policy_ultra
        # If CSP is disabled, no Content-Security-Policy header is added at all

        # Additional CORS headers for better compatibility
        if request.method == "OPTIONS":
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "3600"

        return response