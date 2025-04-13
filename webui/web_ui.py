import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

# If your main API is at a different URL/port, set it here:
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

# Create the FastAPI app for the UI
app = FastAPI(title="Phishing Detection Web UI", version="1.0.0")

# Set up logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("web_ui")
logger.setLevel(logging.INFO)

# Mount the static directory for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Mount your static directory if not already done
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Render the main UI using a Jinja2 template.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "api_base_url": API_BASE_URL
        }
    )


##
# uvicorn web_ui:app --host 0.0.0.0 --port 8000 --reload
##