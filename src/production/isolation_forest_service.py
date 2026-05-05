"""
Production Isolation Forest Service
Real-time fraud detection endpoint
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import joblib
from pathlib import Path

from ..models.isolation_forest import AnomalyDetectionEnsemble
from ..utils import DataScaler


class IsolationForestService:
    """
    Production service for real-time Isolation Forest scoring
    Handles model loading, caching, versioning
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize production service
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.scaler = None
        self.model_version = "1.0"
        self.last_updated = None
        self.inference_count = 0
        self.errors = []
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load pre-trained Isolation Forest model
        
        Args:
            model_path: Path to saved model
        """
        try:
            self.model = joblib.load(model_path)
            self.is_loaded = True
            self.last_updated = datetime.now()
            print(f"✓ Model loaded from {model_path}")
            print(f"  Version: {self.model_version}")
            print(f"  Updated: {self.last_updated}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.errors.append(str(e))
            self.is_loaded = False
    
    def train_and_save(self, X_train: np.ndarray, save_path: str):
        """
        Train new model and save to disk
        
        Args:
            X_train: Training data (normal transactions only)
            save_path: Path to save model
        """
        print(f"Training Isolation Forest...")
        
        self.model = AnomalyDetectionEnsemble()
        self.model.train(X_train)
        
        # Save model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        
        self.is_loaded = True
        self.last_updated = datetime.now()
        print(f"✓ Model trained and saved to {save_path}")
    
    def predict_single(self, transaction: np.ndarray, return_details: bool = False) -> Dict[str, Any]:
        """
        Score a single transaction for fraud
        
        Args:
            transaction: Feature vector (shape: n_features,)
            return_details: Include detailed scoring info
        
        Returns:
            Prediction with score and metadata
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'status': 'FAILED'
            }
        
        start_time = time.time()
        
        try:
            # Reshape for prediction
            X = transaction.reshape(1, -1)
            
            # Get ensemble prediction
            pred_binary = self.model.ensemble_vote(X)[0]
            
            # Get anomaly scores from each model
            scores = self.model.get_anomaly_scores(X)
            
            # Calculate mean score
            mean_score = np.mean([
                scores['isolation_forest'][0],
                scores['lof'][0],
                scores['elliptic'][0]
            ])
            
            # Determine risk level
            if mean_score > 0.7:
                risk_level = "HIGH"
            elif mean_score > 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            latency_ms = (time.time() - start_time) * 1000
            self.inference_count += 1
            
            result = {
                'fraud_prediction': int(pred_binary),
                'fraud_probability': float(mean_score),
                'risk_level': risk_level,
                'latency_ms': float(latency_ms),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'inference_id': f"{self.inference_count}",
                'status': 'SUCCESS'
            }
            
            if return_details:
                result['details'] = {
                    'isolation_forest_score': float(scores['isolation_forest'][0]),
                    'lof_score': float(scores['lof'][0]),
                    'elliptic_score': float(scores['elliptic'][0]),
                }
            
            return result
        
        except Exception as e:
            self.errors.append(str(e))
            return {
                'error': str(e),
                'status': 'FAILED',
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, transactions: np.ndarray, 
                     batch_id: str = None) -> Dict[str, Any]:
        """
        Score batch of transactions
        
        Args:
            transactions: Feature matrix (n_transactions, n_features)
            batch_id: Identifier for batch
        
        Returns:
            Batch predictions with metadata
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded', 'status': 'FAILED'}
        
        start_time = time.time()
        
        try:
            # Get predictions
            pred_binary = self.model.ensemble_vote(transactions)
            scores = self.model.get_anomaly_scores(transactions)
            
            # Calculate mean scores
            mean_scores = np.mean([
                scores['isolation_forest'],
                scores['lof'],
                scores['elliptic']
            ], axis=0)
            
            # Determine risk levels
            risk_levels = np.where(
                mean_scores > 0.7, 'HIGH',
                np.where(mean_scores > 0.5, 'MEDIUM', 'LOW')
            )
            
            # Statistics
            n_high_risk = np.sum(risk_levels == 'HIGH')
            n_medium_risk = np.sum(risk_levels == 'MEDIUM')
            n_fraud = np.sum(pred_binary)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'batch_id': batch_id or f"batch_{int(time.time())}",
                'n_transactions': len(transactions),
                'n_fraud_detected': int(n_fraud),
                'n_high_risk': int(n_high_risk),
                'n_medium_risk': int(n_medium_risk),
                'fraud_rate': float(n_fraud / len(transactions)),
                'latency_ms': float(latency_ms),
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS',
                'predictions': {
                    'fraud_binary': pred_binary.tolist(),
                    'fraud_probability': mean_scores.tolist(),
                    'risk_levels': risk_levels.tolist()
                }
            }
        
        except Exception as e:
            self.errors.append(str(e))
            return {
                'error': str(e),
                'status': 'FAILED',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health metrics
        """
        return {
            'service': 'Isolation Forest Production',
            'status': 'HEALTHY' if self.is_loaded else 'UNHEALTHY',
            'model_loaded': self.is_loaded,
            'model_version': self.model_version,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'inference_count': self.inference_count,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring
        """
        return {
            'total_inferences': self.inference_count,
            'model_version': self.model_version,
            'uptime_since': self.last_updated.isoformat() if self.last_updated else None,
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance for production
_service_instance = None


def get_service(model_path: str = None) -> IsolationForestService:
    """
    Get or create production service instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = IsolationForestService(model_path)
    return _service_instance
