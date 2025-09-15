from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse
import json
import os

docs_router = APIRouter()

@docs_router.get("/openapi.json")
async def get_openapi_json():
    """Get OpenAPI specification in JSON format."""
    from ..main import app
    return JSONResponse(content=app.openapi())

@docs_router.get("/api-docs")
async def get_api_documentation():
    """Get comprehensive API documentation."""
    docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "API_DOCUMENTATION.md")
    if os.path.exists(docs_path):
        return FileResponse(docs_path, media_type="text/markdown", filename="API_DOCUMENTATION.md")
    else:
        return {"error": "Documentation file not found"}

@docs_router.get("/api-spec")
async def get_api_spec_summary():
    """Get a summary of all available endpoints."""
    return {
        "title": "VectorShield API",
        "version": "1.0.0",
        "description": "Modern spam detection engine using vector similarity search",
        "base_url": "http://localhost:5000",
        "endpoints": {
            "analysis": {
                "POST /analyze": "Standard email analysis",
                "POST /analyze/enhanced": "Enhanced analysis with comparison",
                "POST /analyze/ensemble": "Ensemble prediction with multiple models"
            },
            "training": {
                "POST /insert": "Insert labeled training email",
                "POST /report_false_positive": "Remove false positive from database"
            },
            "utilities": {
                "POST /parse_eml": "Parse .eml file format",
                "GET /metrics": "Prometheus metrics for monitoring"
            },
            "documentation": {
                "GET /docs": "Interactive Swagger UI",
                "GET /redoc": "ReDoc documentation",
                "GET /openapi.json": "OpenAPI specification",
                "GET /api-docs": "Comprehensive documentation",
                "GET /api-spec": "API specification summary"
            }
        },
        "example_request": {
            "subject": "Urgent: Verify Your Account",
            "body": "Click here to verify your account immediately.",
            "sender": "security@bank.com",
            "recipient": "user@example.com"
        },
        "example_response": {
            "phishing_score": 85,
            "confidence_level": "High",
            "closest_match": "bad",
            "reasons": ["Enhanced prediction: bad", "Confidence level: High"]
        }
    }