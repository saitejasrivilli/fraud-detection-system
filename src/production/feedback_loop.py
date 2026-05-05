"""
Feedback Loop System
Continuous model improvement through feedback from manual reviews and actual fraud cases
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collect feedback from various sources"""
    
    def __init__(self, feedback_file: str = 'feedback.json'):
        self.feedback_file = Path(feedback_file)
        self.feedback = []
        self.load_feedback()
    
    def add_manual_review_feedback(self, case_id: str, transaction_id: str,
                                  investigator_decision: str, 
                                  model_prediction: bool,
                                  confidence: float,
                                  notes: str = ""):
        """
        Add feedback from manual investigator review
        
        Args:
            case_id: Review case ID
            transaction_id: Transaction ID
            investigator_decision: Ground truth (FRAUD/NOT_FRAUD)
            model_prediction: What model predicted
            confidence: Model confidence (0-1)
            notes: Additional context
        """
        feedback_item = {
            'feedback_type': 'MANUAL_REVIEW',
            'case_id': case_id,
            'transaction_id': transaction_id,
            'ground_truth': investigator_decision,
            'model_prediction': model_prediction,
            'confidence': confidence,
            'is_correct': (
                (investigator_decision == 'FRAUD' and model_prediction) or
                (investigator_decision == 'NOT_FRAUD' and not model_prediction)
            ),
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback.append(feedback_item)
        
        status = "✓ Correct" if feedback_item['is_correct'] else "✗ Wrong"
        logger.info(f"Feedback recorded: {case_id} | {status}")
        
        return feedback_item
    
    def add_system_feedback(self, transaction_id: str, feedback_type: str,
                           feedback_value: float, context: str = ""):
        """
        Add automated feedback from system (e.g., chargeback detection)
        """
        feedback_item = {
            'feedback_type': feedback_type,
            'transaction_id': transaction_id,
            'feedback_value': feedback_value,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback.append(feedback_item)
        logger.info(f"System feedback: {feedback_type} | {transaction_id}")
        
        return feedback_item
    
    def get_recent_feedback(self, hours: int = 24, 
                           feedback_type: str = None) -> List[Dict[str, Any]]:
        """Get recent feedback"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent = [f for f in self.feedback 
                 if datetime.fromisoformat(f['timestamp']) > cutoff_time]
        
        if feedback_type:
            recent = [f for f in recent if f['feedback_type'] == feedback_type]
        
        return recent
    
    def save_feedback(self):
        """Persist feedback to disk"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback, f, indent=2, default=str)
            logger.info(f"✓ Feedback saved ({len(self.feedback)} items)")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def load_feedback(self):
        """Load feedback from disk"""
        if not self.feedback_file.exists():
            logger.info("No saved feedback found. Starting fresh.")
            return
        
        try:
            with open(self.feedback_file, 'r') as f:
                self.feedback = json.load(f)
            logger.info(f"✓ Loaded {len(self.feedback)} feedback items")
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")


class ModelImprover:
    """
    Analyze feedback and suggest model improvements
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback = feedback_collector
    
    def calculate_accuracy_by_risk_level(self) -> Dict[str, float]:
        """
        Calculate model accuracy by risk level
        Identifies which risk levels the model struggles with
        """
        risk_levels = {
            'HIGH': {'total': 0, 'correct': 0},
            'MEDIUM': {'total': 0, 'correct': 0},
            'LOW': {'total': 0, 'correct': 0}
        }
        
        for fb in self.feedback.feedback:
            if fb.get('feedback_type') != 'MANUAL_REVIEW':
                continue
            
            # Estimate risk level from confidence
            confidence = fb.get('confidence', 0.5)
            if confidence > 0.7:
                risk = 'HIGH'
            elif confidence > 0.5:
                risk = 'MEDIUM'
            else:
                risk = 'LOW'
            
            risk_levels[risk]['total'] += 1
            if fb.get('is_correct'):
                risk_levels[risk]['correct'] += 1
        
        # Calculate accuracy
        accuracy = {}
        for risk, counts in risk_levels.items():
            if counts['total'] > 0:
                accuracy[risk] = counts['correct'] / counts['total'] * 100
            else:
                accuracy[risk] = 0.0
        
        return accuracy
    
    def identify_false_positives(self, min_samples: int = 10) -> Dict[str, Any]:
        """
        Identify patterns in false positives
        """
        false_positives = [
            fb for fb in self.feedback.feedback
            if fb.get('feedback_type') == 'MANUAL_REVIEW' and
            fb.get('model_prediction') == True and
            fb.get('ground_truth') == 'NOT_FRAUD'
        ]
        
        if len(false_positives) < min_samples:
            return {
                'count': len(false_positives),
                'insufficient_data': True,
                'recommendation': f"Need {min_samples - len(false_positives)} more FP samples"
            }
        
        # Analyze false positive patterns
        avg_confidence = np.mean([fb.get('confidence', 0) 
                                 for fb in false_positives])
        
        analysis = {
            'count': len(false_positives),
            'false_positive_rate_percent': (
                len(false_positives) / max(len(self.feedback.feedback), 1) * 100
            ),
            'avg_confidence': float(avg_confidence),
            'recommendation': self._fp_recommendation(avg_confidence, len(false_positives))
        }
        
        return analysis
    
    def _fp_recommendation(self, avg_conf: float, count: int) -> str:
        """Recommendation for reducing false positives"""
        if avg_conf > 0.7:
            return "Increase threshold for high-confidence predictions"
        elif avg_conf > 0.5:
            return "Investigate feature importance - may have spurious features"
        else:
            return "Model uncertainty high - ensemble approach or more features needed"
    
    def identify_false_negatives(self, min_samples: int = 5) -> Dict[str, Any]:
        """
        Identify patterns in false negatives (missed fraud)
        """
        false_negatives = [
            fb for fb in self.feedback.feedback
            if fb.get('feedback_type') == 'MANUAL_REVIEW' and
            fb.get('model_prediction') == False and
            fb.get('ground_truth') == 'FRAUD'
        ]
        
        if len(false_negatives) < min_samples:
            return {
                'count': len(false_negatives),
                'insufficient_data': True,
                'recommendation': f"Need {min_samples - len(false_negatives)} more FN samples"
            }
        
        analysis = {
            'count': len(false_negatives),
            'false_negative_rate_percent': (
                len(false_negatives) / 
                max(sum(1 for fb in self.feedback.feedback 
                       if fb.get('ground_truth') == 'FRAUD'), 1) * 100
            ),
            'critical': len(false_negatives) > 5,
            'recommendation': "Review fraud patterns - may indicate new fraud type"
        }
        
        return analysis
    
    def suggest_threshold_adjustment(self) -> Dict[str, Any]:
        """
        Suggest optimal threshold based on feedback
        """
        manual_reviews = [fb for fb in self.feedback.feedback
                         if fb.get('feedback_type') == 'MANUAL_REVIEW']
        
        if len(manual_reviews) < 20:
            return {
                'status': 'INSUFFICIENT_DATA',
                'samples': len(manual_reviews),
                'needed': 20
            }
        
        # Group by confidence threshold
        results = []
        for threshold in np.arange(0.3, 0.9, 0.1):
            predictions_above = [fb for fb in manual_reviews 
                                if fb.get('confidence', 0) >= threshold]
            
            if predictions_above:
                correct = sum(1 for fb in predictions_above if fb.get('is_correct'))
                accuracy = correct / len(predictions_above) * 100
                recall = len([fb for fb in predictions_above 
                            if fb.get('ground_truth') == 'FRAUD']) / max(
                    len([fb for fb in manual_reviews 
                        if fb.get('ground_truth') == 'FRAUD']), 1) * 100
                
                results.append({
                    'threshold': float(threshold),
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1': 2 * (accuracy * recall) / (accuracy + recall + 1e-6)
                })
        
        # Find optimal threshold
        best = max(results, key=lambda x: x['f1'])
        
        return {
            'current_threshold': 0.5,
            'recommended_threshold': best['threshold'],
            'expected_accuracy': best['accuracy'],
            'expected_recall': best['recall'],
            'expected_f1': best['f1']
        }
    
    def generate_improvement_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive improvement report
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'total_feedback_items': len(self.feedback.feedback),
            'accuracy_by_risk': self.calculate_accuracy_by_risk_level(),
            'false_positives': self.identify_false_positives(),
            'false_negatives': self.identify_false_negatives(),
            'threshold_recommendation': self.suggest_threshold_adjustment(),
            'next_actions': self._get_next_actions()
        }
    
    def _get_next_actions(self) -> List[str]:
        """Suggest next improvement actions"""
        actions = [
            "Monitor model performance on new transactions",
            "Retrain model weekly with accumulated feedback",
            "A/B test threshold adjustments",
            "Analyze misclassified cases for new fraud patterns"
        ]
        
        accuracy = self.calculate_accuracy_by_risk_level()
        if accuracy.get('HIGH', 0) < 85:
            actions.insert(0, "Urgent: Improve high-risk detection")
        
        return actions
    
    def export_report(self, filepath: str):
        """Export improvement report"""
        try:
            report = self.generate_improvement_report()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✓ Report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
    
    def print_report(self):
        """Print improvement report to console"""
        report = self.generate_improvement_report()
        
        print("\n" + "="*100)
        print("MODEL IMPROVEMENT REPORT")
        print("="*100)
        print(f"\nGenerated: {report['generated_at']}")
        print(f"Total Feedback: {report['total_feedback_items']}")
        
        print("\n" + "-"*100)
        print("ACCURACY BY RISK LEVEL:")
        print("-"*100)
        for risk, acc in report['accuracy_by_risk'].items():
            status = "✓" if acc > 80 else "✗"
            print(f"  {status} {risk:6} Risk: {acc:6.1f}%")
        
        print("\n" + "-"*100)
        print("FALSE POSITIVES:")
        print("-"*100)
        fp = report['false_positives']
        print(f"  Count: {fp.get('count', 0)}")
        print(f"  Rate: {fp.get('false_positive_rate_percent', 0):.2f}%")
        print(f"  Recommendation: {fp.get('recommendation', 'N/A')}")
        
        print("\n" + "-"*100)
        print("FALSE NEGATIVES:")
        print("-"*100)
        fn = report['false_negatives']
        status = "⚠ CRITICAL" if fn.get('critical') else "✓ Normal"
        print(f"  {status}")
        print(f"  Count: {fn.get('count', 0)}")
        print(f"  Rate: {fn.get('false_negative_rate_percent', 0):.2f}%")
        
        print("\n" + "-"*100)
        print("THRESHOLD RECOMMENDATION:")
        print("-"*100)
        tr = report['threshold_recommendation']
        if 'current_threshold' in tr:
            print(f"  Current: {tr['current_threshold']}")
            print(f"  Recommended: {tr['recommended_threshold']}")
            print(f"  Expected Accuracy: {tr['expected_accuracy']:.1f}%")
            print(f"  Expected Recall: {tr['expected_recall']:.1f}%")
        
        print("\n" + "-"*100)
        print("NEXT ACTIONS:")
        print("-"*100)
        for i, action in enumerate(report['next_actions'], 1):
            print(f"  {i}. {action}")
        
        print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    # Example usage
    collector = FeedbackCollector()
    improver = ModelImprover(collector)
    
    # Simulate feedback
    print("\n" + "="*100)
    print("FEEDBACK LOOP SYSTEM")
    print("="*100 + "\n")
    
    for i in range(50):
        is_fraud = np.random.random() < 0.01
        confidence = np.random.uniform(0.2, 0.95)
        
        # Model was mostly correct
        prediction_correct = (np.random.random() < 0.85)
        model_prediction = is_fraud if prediction_correct else not is_fraud
        
        collector.add_manual_review_feedback(
            case_id=f"CASE_{i:05d}",
            transaction_id=f"TXN_{i:05d}",
            investigator_decision="FRAUD" if is_fraud else "NOT_FRAUD",
            model_prediction=model_prediction,
            confidence=confidence,
            notes=f"Test feedback {i}"
        )
    
    collector.save_feedback()
    improver.print_report()
    improver.export_report('improvement_report.json')
