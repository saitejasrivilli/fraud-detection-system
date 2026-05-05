from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import time
import json
from datetime import datetime
import asyncio
from collections import deque


app = FastAPI(title="Fraud Detection API", version="1.0.0")


class TransactionRequest(BaseModel):
    """Request schema for single transaction scoring"""
    customer_id: str
    merchant_id: str
    amount: float
    timestamp: str
    features: List[float] = None


class TransactionBatch(BaseModel):
    """Request schema for batch transaction scoring"""
    transactions: List[TransactionRequest]


class FraudScoreResponse(BaseModel):
    """Response schema for fraud score"""
    customer_id: str
    fraud_score: float
    fraud_probability: float
    is_fraud: bool
    latency_ms: float
    models_agree: int
    models_total: int
    risk_level: str
    reason: str


class StreamingPipeline:
    """
    Simulated streaming pipeline for real-time fraud detection
    In production, would connect to actual models and message queue
    """
    
    def __init__(self):
        self.transaction_queue = deque(maxlen=1000)
        self.scores_history = deque(maxlen=10000)
        self.alerts = []
        self.is_running = False
    
    def add_transaction(self, transaction: TransactionRequest):
        """Add transaction to processing queue"""
        self.transaction_queue.append({
            'data': transaction,
            'received_at': time.time()
        })
    
    def process_batch(self, transactions: List[TransactionRequest]) -> List[Dict]:
        """Process batch of transactions"""
        results = []
        
        for txn in transactions:
            score = self.score_transaction(txn)
            results.append(score)
        
        return results
    
    def score_transaction(self, transaction: TransactionRequest) -> Dict:
        """
        Score single transaction using ensemble of models
        In production: call actual TF/sklearn models
        """
        start_time = time.time()
        
        # Simulate model ensemble
        # In real implementation: call isolation_forest.predict(), autoencoder.predict(), etc.
        
        if transaction.features is not None:
            features = np.array(transaction.features)
        else:
            # Use transaction data to create features
            features = self._extract_features(transaction)
        
        # Simulate model predictions (in real code: actual model calls)
        scores = {
            'isolation_forest': np.random.random() * 0.3 + (0.7 if transaction.amount > 1000 else 0),
            'autoencoder': np.random.random() * 0.3 + (0.65 if transaction.amount > 1000 else 0),
            'lof': np.random.random() * 0.3 + (0.6 if transaction.amount > 1000 else 0),
            'gcn': np.random.random() * 0.3 + (0.7 if transaction.amount > 1000 else 0),
            'lstm': np.random.random() * 0.3 + (0.5 if transaction.amount > 1000 else 0),
        }
        
        # Ensemble
        mean_score = np.mean(list(scores.values()))
        models_agree = sum(1 for s in scores.values() if s > 0.5)
        
        # Determine risk level
        if mean_score > 0.7:
            risk_level = "HIGH"
            is_fraud = True
        elif mean_score > 0.5:
            risk_level = "MEDIUM"
            is_fraud = True
        elif mean_score > 0.3:
            risk_level = "LOW"
            is_fraud = False
        else:
            risk_level = "MINIMAL"
            is_fraud = False
        
        # Generate reason
        reason = self._generate_reason(transaction, mean_score)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'customer_id': transaction.customer_id,
            'fraud_score': float(mean_score),
            'fraud_probability': float(mean_score),
            'is_fraud': is_fraud,
            'latency_ms': float(latency_ms),
            'models_agree': models_agree,
            'models_total': 5,
            'risk_level': risk_level,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.scores_history.append(result)
        
        # Alert if high risk
        if risk_level == "HIGH":
            self.alerts.append({
                'transaction_id': f"{transaction.customer_id}_{int(time.time())}",
                'risk_level': risk_level,
                'fraud_score': mean_score,
                'timestamp': datetime.now().isoformat()
            })
        
        return result
    
    def _extract_features(self, transaction: TransactionRequest) -> np.ndarray:
        """Extract features from transaction"""
        # Simplified feature extraction
        features = np.array([
            transaction.amount / 1000,  # Normalize
            hash(transaction.merchant_id) % 100 / 100,
            hash(transaction.customer_id) % 100 / 100,
            1.0,  # Placeholder for additional features
            0.5,
            0.3,
            0.7,
            0.4,
            0.6,
            0.2,
        ])
        return features
    
    def _generate_reason(self, transaction: TransactionRequest, score: float) -> str:
        """Generate human-readable reason for fraud score"""
        reasons = []
        
        if transaction.amount > 1000:
            reasons.append("Unusually high amount")
        
        if score > 0.7:
            reasons.append("Multiple models flagged")
        
        if not reasons:
            reasons.append("Routine transaction")
        
        return " | ".join(reasons)
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        recent_scores = list(self.scores_history)
        
        if not recent_scores:
            return {
                'total_transactions': 0,
                'fraud_detections': 0,
                'avg_score': 0,
                'avg_latency_ms': 0,
            }
        
        fraud_count = sum(1 for s in recent_scores if s['is_fraud'])
        
        return {
            'total_transactions': len(recent_scores),
            'fraud_detections': fraud_count,
            'fraud_rate': fraud_count / len(recent_scores),
            'avg_score': np.mean([s['fraud_score'] for s in recent_scores]),
            'avg_latency_ms': np.mean([s['latency_ms'] for s in recent_scores]),
            'max_latency_ms': max([s['latency_ms'] for s in recent_scores]),
            'recent_alerts': len(self.alerts),
        }


# Initialize pipeline
pipeline = StreamingPipeline()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "Fraud Detection API is running"}


@app.post("/score", response_model=FraudScoreResponse)
async def score_transaction(transaction: TransactionRequest):
    """
    Score a single transaction for fraud
    
    Example:
    {
        "customer_id": "CUST123",
        "merchant_id": "MERCH456",
        "amount": 150.50,
        "timestamp": "2024-01-15T10:30:00"
    }
    """
    try:
        result = pipeline.score_transaction(transaction)
        return FraudScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_batch")
async def score_batch(batch: TransactionBatch):
    """
    Score a batch of transactions
    """
    try:
        results = pipeline.process_batch(batch.transactions)
        return {
            'transactions_scored': len(results),
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get pipeline statistics"""
    return pipeline.get_statistics()


@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """Get recent fraud alerts"""
    recent_alerts = list(pipeline.alerts)[-limit:]
    return {
        'total_alerts': len(pipeline.alerts),
        'recent_alerts': recent_alerts
    }


# For development: run with: uvicorn src.streaming:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
