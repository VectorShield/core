import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from ..utils import parse_raw_eml

logger = logging.getLogger("phishing_api")

parse_eml_router = APIRouter()

@parse_eml_router.post("/parse_eml")
async def parse_eml(file: UploadFile = File(...)):
    """
    Parse a raw EML file. Returns a JSON structure with the subject, body (as base64), and sender.
    """
    try:
        eml_bytes = await file.read()
        parsed_email = parse_raw_eml(eml_bytes)
        return {"message": "Parsed EML", "email": parsed_email}
    except Exception as e:
        logger.error(f"Failed to parse EML: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse EML: {e}")
