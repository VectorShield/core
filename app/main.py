from fastapi import FastAPI
from contextlib import asynccontextmanager
from .logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from .config import API_TITLE, API_VERSION
from .routes.metrics import PrometheusMiddleware, metrics_router
from .routes.insert import insert_router
from .database import ensure_collection_exists
from .routes.analyze import analyze_router
from .routes.enhanced_analyze import enhanced_analyze_router
from .routes.report import report_router
from .routes.parse_eml import parse_eml_router
from .routes.docs import docs_router
from .middleware import SecurityHeadersMiddleware
from .batch_upsert import init_batch_upsert
import logging
import sentry_sdk

logger = logging.getLogger("phishing_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs *before* the server starts (like the old on_event("startup")) 
    and *after* the server stops, if needed.
    """
    # BEFORE serving:
    setup_logging()
    sentry_sdk.init(
        dsn="https://62070df56536d744deb5335983041800@o507054.ingest.us.sentry.io/4508866643099648",
        send_default_pii=True,
        traces_sample_rate=None,
    )

    logger.info("Starting up... ensuring Qdrant collection and init batch upsert.")
    await ensure_collection_exists()
    await init_batch_upsert()

    yield  # <-- The application is now running

    # AFTER serving (optional shutdown logic goes here)
    logger.info("Shutting down... (optional teardown)")

# Now pass `lifespan=lifespan` to the app:
app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

# CORS middleware - configured for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for web UI access
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "*",
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "Accept",
        "Origin",
        "User-Agent",
        "Cache-Control",
        "Pragma",
        "X-Custom-Header"
    ],
    expose_headers=["*"],
    max_age=3600  # Cache preflight requests for 1 hour
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Then do your normal setup
app.add_middleware(PrometheusMiddleware)

app.include_router(metrics_router)
app.include_router(insert_router)
app.include_router(analyze_router)
app.include_router(enhanced_analyze_router)
app.include_router(report_router)
app.include_router(parse_eml_router)
app.include_router(docs_router)

logger.info("API started")


# Initialize background batch upsert
#@app.on_event("startup")
#async def startup_event():
#    # FIRST: Make sure the collection exists
#    await ensure_collection_exists()
#    await init_batch_upsert()  # Ensure async function is awaited

### start ! note that the start can take a while
# uvicorn app.main:app --host 0.0.0.0 --port 5000
####