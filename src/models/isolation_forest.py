import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import time
from typing import Dict, Any, Tuple


class AnomalyDetectionEnsemble:
    """Combine multiple anomaly detection methods"""
    
    def __init__(self):
        self.isolation_forest = None
        self.lof = None
        self.elliptic = None
        self.models_trained = False
    
    def train(self, X: np.ndarray):
        """Train all three anomaly detection models"""
        print("\nTraining anomaly detection models...")
        print("-" * 60)
        
        start = time.time()
        
        # Isolation Forest
        print("1. Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=0.01,  # Assume 1% fraud
            random_state=42,
            n_jobs=-1,
            n_estimators=100
        )
        self.isolation_forest.fit(X)
        print("   ✓ Isolation Forest trained")
        
        # Local Outlier Factor
        print("2. Training Local Outlier Factor...")
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.01,
            novelty=True,  # For scoring new data
            n_jobs=-1
        )
        self.lof.fit(X)
        print("   ✓ LOF trained")
        
        # Elliptic Envelope
        print("3. Training Elliptic Envelope...")
        self.elliptic = EllipticEnvelope(
            contamination=0.01,
            random_state=42
        )
        self.elliptic.fit(X)
        print("   ✓ Elliptic Envelope trained")
        
        self.models_trained = True
        elapsed = time.time() - start
        print(f"\nAll models trained in {elapsed:.2f}s")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict anomalies with all models
        Returns dict with predictions for each model
        """
        if not self.models_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        results = {}
        
        # Isolation Forest: -1 = anomaly, 1 = normal
        if_pred = self.isolation_forest.predict(X)
        results['isolation_forest'] = (if_pred == -1).astype(int)
        
        # LOF: -1 = anomaly, 1 = normal
        lof_pred = self.lof.predict(X)
        results['lof'] = (lof_pred == -1).astype(int)
        
        # Elliptic Envelope: -1 = anomaly, 1 = normal
        ee_pred = self.elliptic.predict(X)
        results['elliptic'] = (ee_pred == -1).astype(int)
        
        return results
    
    def get_anomaly_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get continuous anomaly scores (0-1, where 1 = more anomalous)"""
        if not self.models_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        scores = {}
        
        # Isolation Forest: lower scores are more anomalous
        if_scores = self.isolation_forest.score_samples(X)
        # Convert to 0-1 where 1 = anomalous
        scores['isolation_forest'] = 1.0 / (1.0 + np.exp(if_scores))
        
        # LOF: higher scores are more anomalous
        lof_scores = self.lof.score_samples(X)
        scores['lof'] = 1.0 / (1.0 + np.exp(-lof_scores))
        
        # Elliptic Envelope: lower scores are more anomalous
        ee_scores = self.elliptic.score_samples(X)
        scores['elliptic'] = 1.0 / (1.0 + np.exp(ee_scores))
        
        return scores
    
    def ensemble_vote(self, X: np.ndarray, voting_threshold: int = 2) -> np.ndarray:
        """
        Ensemble prediction: flag if 2+ models agree it's an anomaly
        
        Args:
            X: feature matrix
            voting_threshold: number of models that must agree (default: 2/3)
        
        Returns:
            ensemble_predictions: 1 = anomaly, 0 = normal
        """
        predictions = self.predict(X)
        
        # Sum votes
        votes = (
            predictions['isolation_forest'] +
            predictions['lof'] +
            predictions['elliptic']
        )
        
        # Flag if 2+ models agree
        ensemble = (votes >= voting_threshold).astype(int)
        
        return ensemble
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on ground truth labels
        """
        print("\nEvaluating anomaly detection models...")
        print("-" * 60)
        
        predictions = self.predict(X)
        scores = self.get_anomaly_scores(X)
        ensemble = self.ensemble_vote(X)
        
        results = {}
        
        # Evaluate each model
        for model_name in ['isolation_forest', 'lof', 'elliptic']:
            pred = predictions[model_name]
            score = scores[model_name]
            
            results[model_name] = {
                'precision': precision_score(y, pred, zero_division=0),
                'recall': recall_score(y, pred, zero_division=0),
                'f1': f1_score(y, pred, zero_division=0),
                'auc_roc': roc_auc_score(y, score),
            }
            
            print(f"\n{model_name.upper()}:")
            print(f"  Precision: {results[model_name]['precision']:.3f}")
            print(f"  Recall:    {results[model_name]['recall']:.3f}")
            print(f"  F1:        {results[model_name]['f1']:.3f}")
            print(f"  AUC-ROC:   {results[model_name]['auc_roc']:.3f}")
        
        # Evaluate ensemble
        ensemble_score = (
            scores['isolation_forest'] +
            scores['lof'] +
            scores['elliptic']
        ) / 3
        
        results['ensemble'] = {
            'precision': precision_score(y, ensemble, zero_division=0),
            'recall': recall_score(y, ensemble, zero_division=0),
            'f1': f1_score(y, ensemble, zero_division=0),
            'auc_roc': roc_auc_score(y, ensemble_score),
        }
        
        print(f"\nENSEMBLE (voting):")
        print(f"  Precision: {results['ensemble']['precision']:.3f}")
        print(f"  Recall:    {results['ensemble']['recall']:.3f}")
        print(f"  F1:        {results['ensemble']['f1']:.3f}")
        print(f"  AUC-ROC:   {results['ensemble']['auc_roc']:.3f}")
        
        return results
    
    def save(self, filepath: str):
        """Save models to disk"""
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'lof': self.lof,
            'elliptic': self.elliptic
        }, filepath)
        print(f"Models saved to {filepath}")
    
    def load(self, filepath: str):
        """Load models from disk"""
        models = joblib.load(filepath)
        self.isolation_forest = models['isolation_forest']
        self.lof = models['lof']
        self.elliptic = models['elliptic']
        self.models_trained = True
        print(f"Models loaded from {filepath}")
