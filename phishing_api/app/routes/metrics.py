import time
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi import APIRouter, Request

# Prometheus metrics
REQUEST_LATENCY = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["method", "path"]
)
REQUEST_COUNT = Counter(
    "request_count",
    "Number of requests by method, path and HTTP status",
    ["method", "path", "status_code"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        request_duration = time.perf_counter() - start_time

        method = request.method
        path = request.url.path
        status_code = response.status_code

        REQUEST_LATENCY.labels(method=method, path=path).observe(request_duration)
        REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()

        return response

metrics_router = APIRouter()

@metrics_router.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
