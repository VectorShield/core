import logging
from .config import LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """
    Configures the logging for the entire application.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("phishing_api")
    logger.setLevel(logging.INFO)

    # Duplicate to uvicorn logs if needed
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = logger.handlers
    uvicorn_logger.setLevel(logging.INFO)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = logger.handlers
    uvicorn_error_logger.setLevel(logging.INFO)

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = logger.handlers
    uvicorn_access_logger.setLevel(logging.INFO)
