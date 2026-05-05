"""
FEEDBACK LOOP FOR MODEL IMPROVEMENT
Collects ground truth, measures model drift, retrains models
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FeedbackLoop')


class FeedbackCollector:
    """
    Collects ground truth feedback from manual reviews and system confirmations
    """
    
    def __init__(self, db_path: str = 'feedback.db'):
        """
        Args:
            db_path: Database for feedback storage
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize feedback database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    prediction_timestamp TEXT,
                    ground_truth_label INTEGER,
                    ground_truth_source TEXT,
                    feedback_timestamp TEXT,
                    model_prediction REAL,
                    model_name TEXT,
                    correct BOOLEAN,
                    confidence REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    model_name TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    sample_count INTEGER,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feedback database initialized")
        except Exception as e:
            logger.error(f"Database init failed: {e}")
    
    def add_feedback(
        self,
        transaction_id: str,
        ground_truth_label: int,
        source: str,
        model_prediction: float,
        model_name: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Add feedback entry (ground truth)
        
        Args:
            transaction_id: Transaction ID
            ground_truth_label: 1=fraud, 0=normal (from human review)
            source: Source of feedback (e.g., 'manual_review', 'chargeback', 'verified_normal')
            model_prediction: Original model prediction
            model_name: Which model made the prediction
            confidence: Confidence in feedback label
        
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if correct
            correct = (model_prediction > 0.5) == (ground_truth_label == 1)
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback
                (transaction_id, prediction_timestamp, ground_truth_label, ground_truth_source,
                 feedback_timestamp, model_prediction, model_name, correct, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction_id,
                datetime.now().isoformat(),
                ground_truth_label,
                source,
                datetime.now().isoformat(),
                model_prediction,
                model_name,
                correct,
                confidence
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(
                f"Feedback added: {transaction_id} | "
                f"Label: {'FRAUD' if ground_truth_label else 'NORMAL'} | "
                f"Correct: {correct}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False
    
    def get_recent_feedback(self, days: int = 7, model_name: str = None) -> pd.DataFrame:
        """
        Get feedback from recent period
        
        Args:
            days: Days to look back
            model_name: Filter by model (None = all models)
        
        Returns:
            DataFrame of feedback
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            if model_name:
                query = f"""
                    SELECT * FROM feedback
                    WHERE feedback_timestamp >= '{cutoff}'
                    AND model_name = '{model_name}'
                    ORDER BY feedback_timestamp DESC
                """
            else:
                query = f"""
                    SELECT * FROM feedback
                    WHERE feedback_timestamp >= '{cutoff}'
                    ORDER BY feedback_timestamp DESC
                """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to retrieve feedback: {e}")
            return pd.DataFrame()
    
    def calculate_model_metrics(self, model_name: str, days: int = 7) -> Dict[str, float]:
        """
        Calculate model performance on feedback data
        
        Args:
            model_name: Model to evaluate
            days: Period to evaluate
        
        Returns:
            Metrics dictionary
        """
        feedback_df = self.get_recent_feedback(days=days, model_name=model_name)
        
        if len(feedback_df) == 0:
            logger.warning(f"No feedback for {model_name}")
            return {}
        
        y_true = feedback_df['ground_truth_label'].values
        y_pred_binary = (feedback_df['model_prediction'] > 0.5).astype(int)
        y_pred_proba = feedback_df['model_prediction'].values
        
        metrics = {
            'sample_count': len(feedback_df),
            'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred_binary, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
            'accuracy': float((y_pred_binary == y_true).mean())
        }
        
        return metrics
    
    def save_metrics_to_db(self, model_name: str, metrics: Dict[str, float]):
        """Save metrics snapshot to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance
                (date, model_name, precision, recall, f1_score, auc_roc, sample_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                model_name,
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('auc_roc', 0),
                metrics.get('sample_count', 0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


class ModelDriftDetector:
    """
    Detects performance drift in production models
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        """
        Args:
            feedback_collector: FeedbackCollector instance
        """
        self.feedback = feedback_collector
        self.drift_threshold = 0.05  # 5% drop in F1
        self.baseline_metrics = {}
    
    def set_baseline(self, model_name: str, baseline_metrics: Dict[str, float]):
        """
        Set baseline metrics for drift detection
        
        Args:
            model_name: Model name
            baseline_metrics: Baseline performance metrics
        """
        self.baseline_metrics[model_name] = baseline_metrics
        logger.info(f"Baseline set for {model_name}: F1={baseline_metrics.get('f1', 0):.3f}")
    
    def check_drift(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Check if model has drifted
        
        Args:
            model_name: Model to check
            days: Period to evaluate
        
        Returns:
            Drift analysis
        """
        current_metrics = self.feedback.calculate_model_metrics(model_name, days=days)
        baseline = self.baseline_metrics.get(model_name, {})
        
        if not baseline or not current_metrics:
            return {'status': 'insufficient_data'}
        
        # Calculate drift
        baseline_f1 = baseline.get('f1', 0)
        current_f1 = current_metrics.get('f1', 0)
        
        f1_drift = (baseline_f1 - current_f1) / max(baseline_f1, 0.01)
        has_drifted = f1_drift > self.drift_threshold
        
        analysis = {
            'model_name': model_name,
            'baseline_f1': float(baseline_f1),
            'current_f1': float(current_f1),
            'f1_drift_pct': float(f1_drift * 100),
            'has_drifted': has_drifted,
            'drift_severity': self._classify_drift(f1_drift),
            'sample_count': current_metrics.get('sample_count', 0),
            'recommendation': self._get_recommendation(has_drifted, f1_drift)
        }
        
        if has_drifted:
            logger.warning(f"DRIFT DETECTED: {model_name} | F1 dropped {f1_drift*100:.1f}%")
        
        return analysis
    
    def _classify_drift(self, drift_pct: float) -> str:
        """Classify drift severity"""
        if drift_pct < 0.02:
            return "minor"
        elif drift_pct < 0.05:
            return "moderate"
        elif drift_pct < 0.10:
            return "significant"
        else:
            return "critical"
    
    def _get_recommendation(self, has_drifted: bool, drift_pct: float) -> str:
        """Get action recommendation"""
        if not has_drifted:
            return "Continue monitoring"
        elif drift_pct < 0.10:
            return "Monitor closely, schedule retraining"
        else:
            return "URGENT: Retrain model immediately"


class RetrainingPipeline:
    """
    Automated retraining pipeline triggered by drift detection
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        """
        Args:
            feedback_collector: FeedbackCollector instance
        """
        self.feedback = feedback_collector
        self.retraining_history = []
    
    def prepare_retraining_data(self, model_name: str, days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for retraining
        
        Args:
            model_name: Model to retrain
            days: Days of feedback to use
        
        Returns:
            (X, y) training data
        """
        feedback_df = self.feedback.get_recent_feedback(days=days, model_name=model_name)
        
        if len(feedback_df) == 0:
            logger.warning("No data for retraining")
            return None, None
        
        # Features: typically stored with feedback, but for this example:
        # We use model prediction confidence as proxy
        X = feedback_df[['model_prediction']].values
        y = feedback_df['ground_truth_label'].values
        
        logger.info(f"Prepared {len(X)} samples for retraining {model_name}")
        
        return X, y
    
    def should_retrain(self, model_name: str) -> bool:
        """
        Determine if retraining is needed
        
        Args:
            model_name: Model to check
        
        Returns:
            True if retraining recommended
        """
        # Check feedback volume
        feedback_df = self.feedback.get_recent_feedback(days=7, model_name=model_name)
        
        min_feedback_for_retrain = 100
        
        if len(feedback_df) < min_feedback_for_retrain:
            return False
        
        # Check for performance degradation
        current_metrics = self.feedback.calculate_model_metrics(model_name, days=7)
        
        if current_metrics.get('f1', 0) < 0.7:
            return True
        
        return False
    
    def log_retraining(self, model_name: str, reason: str, metrics: Dict[str, float]):
        """Log retraining event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'reason': reason,
            'metrics': metrics
        }
        
        self.retraining_history.append(event)
        logger.info(f"Retraining logged for {model_name}: {reason}")
    
    def get_retraining_schedule(self) -> List[Dict[str, Any]]:
        """Get recommended retraining schedule"""
        schedule = []
        
        # Example: Retrain IF weekly, GCN monthly
        schedule.append({
            'model': 'isolation_forest',
            'frequency': 'weekly',
            'trigger': 'performance drift',
            'last_retrained': datetime.now().isoformat()
        })
        
        schedule.append({
            'model': 'autoencoder',
            'frequency': 'biweekly',
            'trigger': 'drift or > 1000 feedback samples',
            'last_retrained': datetime.now().isoformat()
        })
        
        schedule.append({
            'model': 'gcn',
            'frequency': 'monthly',
            'trigger': 'new fraud ring patterns',
            'last_retrained': datetime.now().isoformat()
        })
        
        return schedule


# Automated feedback loop orchestration
class FeedbackLoopOrchestrator:
    """
    Orchestrates the complete feedback loop
    """
    
    def __init__(self):
        """Initialize orchestrator"""
        self.feedback_collector = FeedbackCollector()
        self.drift_detector = ModelDriftDetector(self.feedback_collector)
        self.retraining_pipeline = RetrainingPipeline(self.feedback_collector)
        self.models = ['isolation_forest', 'autoencoder', 'lstm', 'gcn']
    
    def run_feedback_loop(self) -> Dict[str, Any]:
        """
        Execute complete feedback loop cycle
        
        Returns:
            Loop results and recommendations
        """
        logger.info("Starting feedback loop cycle...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in self.models:
            logger.info(f"Processing {model_name}...")
            
            # Get current metrics
            metrics = self.feedback_collector.calculate_model_metrics(model_name, days=7)
            
            if not metrics:
                logger.warning(f"No metrics for {model_name}")
                continue
            
            # Check for drift
            drift_analysis = self.drift_detector.check_drift(model_name, days=7)
            
            # Check if retraining needed
            needs_retrain = self.retraining_pipeline.should_retrain(model_name)
            
            results['models'][model_name] = {
                'current_metrics': metrics,
                'drift_analysis': drift_analysis,
                'needs_retraining': needs_retrain,
                'recommendation': drift_analysis.get('recommendation', 'unknown')
            }
            
            # Log metrics
            self.feedback_collector.save_metrics_to_db(model_name, metrics)
            
            if needs_retrain:
                logger.info(f"⚠️  {model_name} requires retraining")
                self.retraining_pipeline.log_retraining(
                    model_name,
                    f"Performance drift or insufficient data",
                    metrics
                )
        
        logger.info("Feedback loop cycle complete")
        return results
    
    def export_report(self, output_path: str = "feedback_report.json"):
        """Export feedback loop report"""
        results = self.run_feedback_loop()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Report exported to {output_path}")
        return results


if __name__ == "__main__":
    # Example usage
    orchestrator = FeedbackLoopOrchestrator()
    
    # Simulate feedback
    print("Adding sample feedback...")
    for i in range(10):
        orchestrator.feedback_collector.add_feedback(
            transaction_id=f"TXN_{i:04d}",
            ground_truth_label=np.random.randint(0, 2),
            source="manual_review",
            model_prediction=np.random.random(),
            model_name="isolation_forest"
        )
    
    # Run feedback loop
    print("\nRunning feedback loop...")
    results = orchestrator.run_feedback_loop()
    
    print("\n📊 FEEDBACK LOOP RESULTS:")
    print(json.dumps(results, indent=2, default=str))
