from fastapi import FastAPI
from contextlib import asynccontextmanager
from .logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from .config import API_TITLE, API_VERSION
from .routes.metrics import PrometheusMiddleware, metrics_router
from .routes.insert import insert_router
from .database import ensure_collection_exists
from .routes.analyze import analyze_router
from .routes.report import report_router
from .routes.parse_eml import parse_eml_router
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify e.g. ["http://localhost:8000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Then do your normal setup
app.add_middleware(PrometheusMiddleware)

app.include_router(metrics_router)
app.include_router(insert_router)
app.include_router(analyze_router)
app.include_router(report_router)
app.include_router(parse_eml_router)

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