import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .logging_config import setup_logging
from .config import API_TITLE, API_VERSION
from .routes.metrics import PrometheusMiddleware, metrics_router
from .routes.insert import insert_router
from .routes.analyze import analyze_router
from .routes.report import report_router
from .routes.parse_eml import parse_eml_router
from .batch_upsert import init_batch_upsert

# Configure logging on startup
setup_logging()

logger = logging.getLogger("phishing_api")

# Create FastAPI application
app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify e.g. ["http://localhost:8000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include route modules
app.include_router(metrics_router)
app.include_router(insert_router)
app.include_router(analyze_router)
app.include_router(report_router)
app.include_router(parse_eml_router)

logger.info("API started")

# Initialize background batch upsert
@app.on_event("startup")
async def startup_event():
    await init_batch_upsert()  # Ensure async function is awaited

### start ! note that the start can take a while
# uvicorn app.main:app --host 0.0.0.0 --port 5000
####