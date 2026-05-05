"""
FastAPI Production Application
Full integration with database, authentication, and monitoring
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from typing import Optional

# Import custom modules
from src.production import get_orchestrator
from DEPLOYMENT_FILES.auth import (
    Auth, get_current_active_user, require_role, Token, TokenData
)
from DEPLOYMENT_FILES.models import (
    get_db, create_tables, TransactionDAO, FraudCase
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Fraud Detection System",
    description="Production-ready fraud detection API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    create_tables()
    logger.info("✓ Database initialized")
    logger.info("✓ Application started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("✓ Application shutdown")


# ========== SCHEMAS ==========

class TransactionScore(BaseModel):
    """Transaction scoring request"""
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    features: list

class TransactionResponse(BaseModel):
    """Transaction response"""
    transaction_id: str
    fraud_prediction: bool
    fraud_probability: float
    risk_level: str
    latency_ms: float


# ========== ENDPOINTS ==========

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness check"""
    return {"status": "ready"}


# ========== AUTHENTICATION ENDPOINTS ==========

@app.post("/login", response_model=Token)
async def login(username: str, password: str):
    """Login endpoint"""
    token = await Auth.login(username, password)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    return token


@app.post("/token", response_model=Token)
async def get_token(username: str, password: str):
    """Get access token"""
    return await login(username, password)


@app.get("/profile")
async def get_profile(current_user: TokenData = Depends(get_current_active_user)):
    """Get current user profile"""
    return {
        "username": current_user.username,
        "role": current_user.role
    }


# ========== FRAUD DETECTION ENDPOINTS ==========

@app.post("/score", response_model=TransactionResponse)
async def score_transaction(
    request: TransactionScore,
    current_user: TokenData = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Score a transaction"""
    orchestrator = get_orchestrator()
    
    # Score transaction
    result = orchestrator.score_transaction(
        features=request.features,
        transaction_id=request.transaction_id
    )
    
    # Save to database
    TransactionDAO.create(db, {
        "transaction_id": request.transaction_id,
        "customer_id": request.customer_id,
        "merchant_id": request.merchant_id,
        "amount": request.amount,
        "features": request.features,
        "ensemble_score": result['fraud_probability'],
        "fraud_prediction": result['fraud_prediction'],
        "risk_level": result['risk_level']
    })
    
    logger.info(f"✓ Scored {request.transaction_id}: {result['risk_level']}")
    
    return TransactionResponse(
        transaction_id=request.transaction_id,
        fraud_prediction=result['fraud_prediction'],
        fraud_probability=result['fraud_probability'],
        risk_level=result['risk_level'],
        latency_ms=result['latency_ms']
    )


@app.get("/pending-reviews")
async def get_pending_reviews(
    limit: int = 10,
    current_user: TokenData = Depends(require_role("reviewer", "admin")),
    db = Depends(get_db)
):
    """Get pending review cases"""
    cases = TransactionDAO.get_pending_review(db, limit)
    return {
        "count": len(cases),
        "cases": [{"transaction_id": c.transaction_id, "risk_level": c.risk_level} for c in cases]
    }


@app.post("/review/{transaction_id}")
async def review_transaction(
    transaction_id: str,
    decision: str,
    notes: str = None,
    current_user: TokenData = Depends(require_role("reviewer", "admin")),
    db = Depends(get_db)
):
    """Submit fraud review"""
    if decision not in ["APPROVED", "REJECTED"]:
        raise HTTPException(status_code=400, detail="Invalid decision")
    
    txn = TransactionDAO.get_by_id(db, transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Update transaction
    TransactionDAO.update_scores(db, transaction_id, {
        "reviewed": True,
        "review_decision": decision,
        "reviewer_id": current_user.username,
        "review_notes": notes
    })
    
    logger.info(f"✓ {current_user.username} reviewed {transaction_id}: {decision}")
    
    return {
        "transaction_id": transaction_id,
        "decision": decision,
        "reviewer": current_user.username
    }


# ========== MONITORING ENDPOINTS ==========

@app.get("/dashboard")
async def get_dashboard(current_user: TokenData = Depends(get_current_active_user)):
    """Get monitoring dashboard"""
    orchestrator = get_orchestrator()
    return orchestrator.get_dashboard()


@app.get("/health-status")
async def get_health_status(current_user: TokenData = Depends(get_current_active_user)):
    """Get system health"""
    orchestrator = get_orchestrator()
    return orchestrator.get_system_health()


# ========== ADMIN ENDPOINTS ==========

@app.post("/retrain")
async def trigger_retraining(
    reason: str = "Manual trigger",
    current_user: TokenData = Depends(require_role("admin"))
):
    """Trigger model retraining"""
    orchestrator = get_orchestrator()
    job_id = orchestrator.trigger_retraining(reason)
    
    logger.info(f"✓ Retraining triggered by {current_user.username}")
    
    return {
        "job_id": job_id,
        "status": "scheduled",
        "triggered_by": current_user.username
    }


@app.get("/feedback-metrics")
async def get_feedback_metrics(
    current_user: TokenData = Depends(require_role("analyst", "admin"))
):
    """Get model feedback metrics"""
    orchestrator = get_orchestrator()
    return orchestrator.get_feedback_metrics()


# ========== ERROR HANDLING ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
