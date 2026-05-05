"""
PRODUCTION ISOLATION FOREST DEPLOYMENT
Real-time fraud detection for sub-50ms latency requirements
"""

import numpy as np
import joblib
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
from collections import deque
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IFProduction')


class ProductionIsolationForest:
    """
    Production-grade Isolation Forest deployment
    Sub-50ms latency, continuous monitoring, graceful degradation
    """
    
    def __init__(self, model_path: str = None, max_history: int = 10000):
        """
        Args:
            model_path: Path to saved model
            max_history: Size of scoring history buffer
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.is_loaded = False
        
        # Metrics
        self.scoring_history = deque(maxlen=max_history)
        self.latency_buffer = deque(maxlen=1000)
        self.decision_counts = {
            'normal': 0,
            'fraud': 0,
            'errors': 0
        }
        self.start_time = datetime.now()
        
        # Configuration
        self.decision_threshold = 0.5
        self.latency_threshold_ms = 50  # SLA
        self.alert_thresholds = {
            'fraud_rate': 0.05,  # Alert if > 5% fraud in last 100
            'latency_p99': 100,  # Alert if p99 latency > 100ms
            'error_rate': 0.01   # Alert if > 1% errors
        }
        
        # Thread safety
        self.lock = threading.Lock()
    
    def load_model(self, model_path: str = None):
        """Load Isolation Forest model from disk"""
        path = model_path or self.model_path
        if not path:
            raise ValueError("No model path provided")
        
        try:
            logger.info(f"Loading model from {path}")
            self.model = joblib.load(path)
            self.is_loaded = True
            logger.info("✓ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def score_transaction(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Score single transaction with latency tracking
        
        Args:
            features: Feature vector (shape: n_features,)
        
        Returns:
            Decision with confidence and metadata
        """
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise ValueError("Model not loaded")
            
            # Ensure proper shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features)[0]
            score = self.model.score_samples(features)[0]
            
            # Convert to fraud probability (0-1)
            fraud_prob = 1.0 / (1.0 + np.exp(score))
            is_fraud = prediction == -1  # -1 = anomaly/fraud
            
            # Latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Record
            with self.lock:
                self.latency_buffer.append(latency_ms)
                decision = 'fraud' if is_fraud else 'normal'
                self.decision_counts[decision] += 1
            
            # Alert if latency SLA breached
            if latency_ms > self.latency_threshold_ms:
                logger.warning(
                    f"Latency SLA breach: {latency_ms:.2f}ms > {self.latency_threshold_ms}ms"
                )
            
            result = {
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'score': float(score),
                'latency_ms': float(latency_ms),
                'sla_met': latency_ms <= self.latency_threshold_ms,
                'timestamp': datetime.now().isoformat(),
                'decision': decision
            }
            
            # Store in history
            with self.lock:
                self.scoring_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            with self.lock:
                self.decision_counts['errors'] += 1
            
            return {
                'is_fraud': False,  # Fail-safe: don't block
                'fraud_probability': 0.0,
                'score': 0.0,
                'latency_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def score_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Score batch of transactions efficiently
        
        Args:
            features: Feature matrix (shape: n_samples, n_features)
        
        Returns:
            List of decisions
        """
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise ValueError("Model not loaded")
            
            # Batch predict
            predictions = self.model.predict(features)
            scores = self.model.score_samples(features)
            
            # Convert to probabilities
            fraud_probs = 1.0 / (1.0 + np.exp(scores))
            is_frauds = predictions == -1
            
            # Latency
            batch_latency_ms = (time.time() - start_time) * 1000
            per_sample_latency = batch_latency_ms / len(features)
            
            results = []
            for i, (fraud_prob, is_fraud, score) in enumerate(
                zip(fraud_probs, is_frauds, scores)
            ):
                result = {
                    'is_fraud': bool(is_fraud),
                    'fraud_probability': float(fraud_prob),
                    'score': float(score),
                    'latency_ms': float(per_sample_latency),
                    'sla_met': per_sample_latency <= self.latency_threshold_ms,
                    'timestamp': datetime.now().isoformat(),
                    'decision': 'fraud' if is_fraud else 'normal'
                }
                results.append(result)
                
                # Update counts
                with self.lock:
                    self.decision_counts['fraud' if is_fraud else 'normal'] += 1
                    self.latency_buffer.append(per_sample_latency)
                    self.scoring_history.append(result)
            
            logger.info(
                f"Batch scored: {len(features)} samples in {batch_latency_ms:.2f}ms "
                f"({per_sample_latency:.2f}ms per sample)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch scoring error: {e}")
            with self.lock:
                self.decision_counts['errors'] += 1
            
            return [
                {
                    'is_fraud': False,
                    'fraud_probability': 0.0,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                for _ in range(len(features))
            ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics and health status"""
        with self.lock:
            history = list(self.scoring_history)
            latencies = list(self.latency_buffer)
        
        if not history:
            return {'status': 'no_data'}
        
        # Calculate metrics
        total_scored = self.decision_counts['fraud'] + self.decision_counts['normal']
        fraud_count = self.decision_counts['fraud']
        error_count = self.decision_counts['errors']
        
        fraud_rate = fraud_count / max(total_scored, 1)
        error_rate = error_count / max(total_scored + error_count, 1)
        
        # Latency percentiles
        latencies_sorted = sorted(latencies) if latencies else [0]
        p50_latency = np.percentile(latencies_sorted, 50)
        p95_latency = np.percentile(latencies_sorted, 95)
        p99_latency = np.percentile(latencies_sorted, 99)
        
        # Alerts
        alerts = []
        if fraud_rate > self.alert_thresholds['fraud_rate']:
            alerts.append(f"High fraud rate: {fraud_rate:.2%}")
        if p99_latency > self.alert_thresholds['latency_p99']:
            alerts.append(f"High p99 latency: {p99_latency:.2f}ms")
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {error_rate:.2%}")
        
        return {
            'status': 'healthy' if not alerts else 'degraded',
            'alerts': alerts,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_scored': total_scored,
            'fraud_detections': fraud_count,
            'fraud_rate': fraud_rate,
            'error_count': error_count,
            'error_rate': error_rate,
            'latency': {
                'p50_ms': float(p50_latency),
                'p95_ms': float(p95_latency),
                'p99_ms': float(p99_latency),
                'max_ms': float(max(latencies_sorted)),
                'mean_ms': float(np.mean(latencies_sorted))
            },
            'sla_compliance': float(
                sum(1 for l in latencies if l <= self.latency_threshold_ms) / 
                max(len(latencies), 1)
            )
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Quick health check for load balancer"""
        return {
            'healthy': self.is_loaded,
            'model_loaded': self.is_loaded,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON for monitoring"""
        metrics = self.get_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {filepath}")
    
    def reset_metrics(self):
        """Reset metrics (e.g., for daily reporting)"""
        with self.lock:
            self.decision_counts = {'normal': 0, 'fraud': 0, 'errors': 0}
            self.latency_buffer.clear()
            self.scoring_history.clear()
            self.start_time = datetime.now()
        logger.info("Metrics reset")


class IsolationForestPool:
    """
    Model pool for parallel scoring with failover
    Handles multiple model versions/instances
    """
    
    def __init__(self, n_instances: int = 3):
        """
        Args:
            n_instances: Number of model instances to maintain
        """
        self.instances = [
            ProductionIsolationForest() 
            for _ in range(n_instances)
        ]
        self.current_idx = 0
        self.lock = threading.Lock()
    
    def score_transaction(self, features: np.ndarray) -> Dict[str, Any]:
        """Score with automatic failover"""
        for i in range(len(self.instances)):
            idx = (self.current_idx + i) % len(self.instances)
            instance = self.instances[idx]
            
            try:
                result = instance.score_transaction(features)
                if 'error' not in result:
                    with self.lock:
                        self.current_idx = idx
                    return result
            except Exception as e:
                logger.warning(f"Instance {idx} failed: {e}")
                continue
        
        # All instances failed
        logger.error("All instances failed")
        return {
            'is_fraud': False,
            'error': 'All instances failed',
            'timestamp': datetime.now().isoformat()
        }
    
    def load_model(self, model_path: str):
        """Load model to all instances"""
        for instance in self.instances:
            instance.load_model(model_path)
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get metrics from all instances"""
        return {
            f'instance_{i}': inst.get_metrics()
            for i, inst in enumerate(self.instances)
        }


if __name__ == "__main__":
    # Example usage
    print("Production Isolation Forest Module")
    print("For use with FastAPI in src/streaming.py")
