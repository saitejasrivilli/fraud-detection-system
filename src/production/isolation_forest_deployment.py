"""
Production Deployment: Isolation Forest Real-Time Serving
Handles low-latency transaction scoring in production environment
"""

import numpy as np
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
from pathlib import Path


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IsolationForestDeployment:
    """
    Production-grade Isolation Forest model deployment
    Handles model loading, caching, and real-time predictions
    """
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to saved model pickle
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.metadata = {
            'deployed_at': None,
            'model_version': '1.0.0',
            'predictions_made': 0,
            'total_latency_ms': 0,
            'errors': 0
        }
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def load_model(self, model_path: str = None) -> bool:
        """Load pre-trained model from disk"""
        try:
            path = model_path or self.model_path
            if not path:
                logger.warning("No model path provided. Using dummy model.")
                return False
                
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.metadata['deployed_at'] = datetime.now().isoformat()
            logger.info(f"✓ Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.metadata['errors'] += 1
            return False
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for model input"""
        if self.scaler:
            return self.scaler.transform(features.reshape(1, -1))[0]
        return features
    
    def predict(self, transaction_id: str, features: np.ndarray) -> Dict[str, Any]:
        """
        Real-time prediction with latency tracking
        
        Args:
            transaction_id: Unique transaction identifier
            features: Feature vector (10 dimensions)
        
        Returns:
            Prediction with metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if transaction_id in self.prediction_cache:
                cached_pred, cached_time = self.prediction_cache[transaction_id]
                if time.time() - cached_time < self.cache_ttl:
                    logger.debug(f"Cache hit: {transaction_id}")
                    return cached_pred
            
            # Preprocess
            processed_features = self.preprocess(features)
            
            # Predict
            if self.model is None:
                # Fallback: simple threshold on amount
                is_fraud = features[0] > 3.0  # Simple heuristic
                fraud_probability = float(features[0]) / 10.0
            else:
                prediction = self.model.predict([processed_features])[0]
                is_fraud = (prediction == -1)  # -1 = anomaly
                fraud_probability = max(0.0, min(1.0, float(prediction) / 10.0))
            
            # Determine risk level
            if fraud_probability > 0.7:
                risk_level = "HIGH"
            elif fraud_probability > 0.5:
                risk_level = "MEDIUM"
            elif fraud_probability > 0.3:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = {
                'transaction_id': transaction_id,
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_probability),
                'risk_level': risk_level,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.metadata['model_version']
            }
            
            # Cache result
            self.prediction_cache[transaction_id] = (result, time.time())
            
            # Update metadata
            self.metadata['predictions_made'] += 1
            self.metadata['total_latency_ms'] += latency_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {transaction_id}: {e}")
            self.metadata['errors'] += 1
            
            return {
                'transaction_id': transaction_id,
                'is_fraud': False,  # Default to safe (allow transaction)
                'fraud_probability': 0.0,
                'risk_level': 'ERROR',
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def batch_predict(self, transactions: list) -> list:
        """
        Batch prediction for multiple transactions
        
        Args:
            transactions: List of {id, features} dicts
        
        Returns:
            List of predictions
        """
        results = []
        
        for txn in transactions:
            result = self.predict(txn['id'], np.array(txn['features']))
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        avg_latency = (
            self.metadata['total_latency_ms'] / max(self.metadata['predictions_made'], 1)
        )
        
        return {
            'deployed_at': self.metadata['deployed_at'],
            'model_version': self.metadata['model_version'],
            'total_predictions': self.metadata['predictions_made'],
            'avg_latency_ms': avg_latency,
            'total_errors': self.metadata['errors'],
            'error_rate': (
                self.metadata['errors'] / max(self.metadata['predictions_made'], 1) * 100
            ),
            'cache_size': len(self.prediction_cache)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"✓ Checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def health_check(self) -> bool:
        """Check deployment health"""
        stats = self.get_statistics()
        
        # Check if error rate < 1%
        if stats['error_rate'] > 1.0:
            logger.warning(f"High error rate: {stats['error_rate']:.2f}%")
            return False
        
        # Check if model is loaded
        if self.model is None:
            logger.warning("Model not loaded")
            return False
        
        # Check latency < 50ms (SLA)
        if stats['avg_latency_ms'] > 50:
            logger.warning(f"Latency exceeds SLA: {stats['avg_latency_ms']:.2f}ms")
            return False
        
        logger.info("✓ Health check passed")
        return True


class ProductionMetrics:
    """Track production performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_transactions': 0,
            'flagged_as_fraud': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'errors': 0,
            'p99_latency': 0,
            'uptime_percent': 100.0
        }
        self.latencies = []
        self.hourly_volume = {}
    
    def record_prediction(self, prediction: Dict[str, Any]):
        """Record prediction for metrics"""
        self.metrics['total_transactions'] += 1
        
        if prediction.get('is_fraud'):
            self.metrics['flagged_as_fraud'] += 1
        
        risk_level = prediction.get('risk_level', 'UNKNOWN')
        if risk_level == 'HIGH':
            self.metrics['high_risk'] += 1
        elif risk_level == 'MEDIUM':
            self.metrics['medium_risk'] += 1
        elif risk_level == 'LOW':
            self.metrics['low_risk'] += 1
        
        if 'error' in prediction:
            self.metrics['errors'] += 1
        
        self.latencies.append(prediction.get('latency_ms', 0))
        
        # Calculate p99 latency
        if len(self.latencies) > 0:
            self.metrics['p99_latency'] = np.percentile(self.latencies, 99)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        fraud_rate = (
            self.metrics['flagged_as_fraud'] / max(self.metrics['total_transactions'], 1) * 100
        )
        
        return {
            'total_transactions': self.metrics['total_transactions'],
            'fraud_rate_percent': fraud_rate,
            'risk_distribution': {
                'high': self.metrics['high_risk'],
                'medium': self.metrics['medium_risk'],
                'low': self.metrics['low_risk']
            },
            'error_rate_percent': (
                self.metrics['errors'] / max(self.metrics['total_transactions'], 1) * 100
            ),
            'p99_latency_ms': self.metrics['p99_latency'],
            'sla_met': self.metrics['p99_latency'] < 50
        }


if __name__ == "__main__":
    # Example usage
    deployment = IsolationForestDeployment()
    
    # Simulate transactions
    print("\n" + "="*80)
    print("ISOLATION FOREST PRODUCTION DEPLOYMENT")
    print("="*80 + "\n")
    
    for i in range(5):
        features = np.random.randn(10)
        result = deployment.predict(f"TXN_{i:05d}", features)
        print(f"Transaction {i}: Risk={result['risk_level']}, "
              f"Fraud={result['is_fraud']}, Latency={result['latency_ms']:.2f}ms")
    
    print("\n" + "-"*80)
    print("DEPLOYMENT STATISTICS:")
    stats = deployment.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "-"*80)
    print(f"Health Check: {'✓ PASS' if deployment.health_check() else '✗ FAIL'}")
    print("="*80 + "\n")
