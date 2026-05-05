"""
Ensemble Manual Review Queue
Manages high-confidence fraud cases for manual investigator review
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaseStatus(Enum):
    """Status of a review case"""
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class FraudDecision(Enum):
    """Final fraud decision"""
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class ReviewCase:
    """A fraud case awaiting manual review"""
    case_id: str
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    timestamp: str
    
    # Ensemble predictions
    isolation_forest_score: float
    autoencoder_score: float
    lstm_score: float
    gcn_score: float
    ensemble_score: float
    models_agreed: int
    
    # Risk assessment
    risk_level: str
    priority_score: float
    
    # Case management
    status: str = CaseStatus.PENDING.value
    assigned_investigator: Optional[str] = None
    created_at: str = None
    updated_at: str = None
    decision: Optional[str] = None
    decision_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class ManualReviewQueue:
    """
    Manages ensemble-based fraud cases for manual review
    High precision (only high-confidence cases) but thorough investigation
    """
    
    def __init__(self, queue_file: str = 'review_queue.json'):
        """
        Args:
            queue_file: File to persist queue
        """
        self.queue_file = Path(queue_file)
        self.cases: Dict[str, ReviewCase] = {}
        self.statistics = {
            'total_cases': 0,
            'resolved_cases': 0,
            'confirmed_fraud': 0,
            'false_positives': 0,
            'avg_resolution_time_hours': 0,
            'precision': 0.0  # Of manual reviews
        }
        self.load_queue()
    
    def add_case(self, transaction_id: str, customer_id: str, merchant_id: str,
                amount: float, predictions: Dict[str, float]) -> ReviewCase:
        """
        Add new case to review queue
        
        Args:
            transaction_id: Transaction ID
            customer_id: Customer ID
            merchant_id: Merchant ID
            amount: Transaction amount
            predictions: Dict with model scores and ensemble result
        
        Returns:
            Created ReviewCase
        """
        # Calculate ensemble priority
        models_agreed = sum(1 for score in predictions.values() 
                          if isinstance(score, float) and score > 0.5)
        
        ensemble_score = predictions.get('ensemble_score', 0.5)
        
        # Priority: higher ensemble score + more models agreed
        priority_score = (ensemble_score * 0.6 + 
                         (models_agreed / 5) * 0.4 * 100)
        
        # Risk level
        if ensemble_score > 0.8:
            risk_level = "CRITICAL"
        elif ensemble_score > 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"
        
        case_id = f"CASE_{int(datetime.now().timestamp())}_{transaction_id[:8]}"
        
        case = ReviewCase(
            case_id=case_id,
            transaction_id=transaction_id,
            customer_id=customer_id,
            merchant_id=merchant_id,
            amount=amount,
            timestamp=datetime.now().isoformat(),
            isolation_forest_score=predictions.get('isolation_forest', 0.0),
            autoencoder_score=predictions.get('autoencoder', 0.0),
            lstm_score=predictions.get('lstm', 0.0),
            gcn_score=predictions.get('gcn', 0.0),
            ensemble_score=ensemble_score,
            models_agreed=models_agreed,
            risk_level=risk_level,
            priority_score=priority_score
        )
        
        self.cases[case_id] = case
        self.statistics['total_cases'] += 1
        
        logger.info(f"✓ Case added: {case_id} | Risk: {risk_level} | "
                   f"Priority: {priority_score:.2f}")
        
        return case
    
    def assign_case(self, case_id: str, investigator: str) -> bool:
        """
        Assign case to investigator
        """
        if case_id not in self.cases:
            logger.error(f"Case not found: {case_id}")
            return False
        
        case = self.cases[case_id]
        case.assigned_investigator = investigator
        case.status = CaseStatus.IN_REVIEW.value
        case.updated_at = datetime.now().isoformat()
        
        logger.info(f"✓ Case {case_id} assigned to {investigator}")
        
        return True
    
    def resolve_case(self, case_id: str, decision: str, 
                    notes: str = "") -> bool:
        """
        Resolve a case with final decision
        
        Args:
            case_id: Case ID
            decision: One of CONFIRMED_FRAUD, FALSE_POSITIVE, NEEDS_MORE_DATA
            notes: Investigator notes
        """
        if case_id not in self.cases:
            logger.error(f"Case not found: {case_id}")
            return False
        
        if decision not in [d.value for d in FraudDecision]:
            logger.error(f"Invalid decision: {decision}")
            return False
        
        case = self.cases[case_id]
        case.decision = decision
        case.decision_notes = notes
        case.status = CaseStatus.RESOLVED.value
        case.updated_at = datetime.now().isoformat()
        
        # Update statistics
        self.statistics['resolved_cases'] += 1
        
        if decision == FraudDecision.CONFIRMED_FRAUD.value:
            self.statistics['confirmed_fraud'] += 1
        elif decision == FraudDecision.FALSE_POSITIVE.value:
            self.statistics['false_positives'] += 1
        
        # Calculate precision
        resolved = (self.statistics['confirmed_fraud'] + 
                   self.statistics['false_positives'])
        if resolved > 0:
            self.statistics['precision'] = (
                self.statistics['confirmed_fraud'] / resolved * 100
            )
        
        logger.info(f"✓ Case {case_id} resolved: {decision} | "
                   f"Precision: {self.statistics['precision']:.1f}%")
        
        return True
    
    def escalate_case(self, case_id: str, reason: str = "") -> bool:
        """
        Escalate case to higher priority
        """
        if case_id not in self.cases:
            return False
        
        case = self.cases[case_id]
        case.status = CaseStatus.ESCALATED.value
        case.priority_score *= 1.5  # Increase priority
        case.updated_at = datetime.now().isoformat()
        
        if reason:
            if case.decision_notes:
                case.decision_notes += f"\nEscalation: {reason}"
            else:
                case.decision_notes = f"Escalation: {reason}"
        
        logger.info(f"✓ Case {case_id} escalated")
        
        return True
    
    def get_pending_cases(self, limit: int = 10) -> List[ReviewCase]:
        """
        Get pending cases sorted by priority (highest first)
        """
        pending = [case for case in self.cases.values() 
                  if case.status == CaseStatus.PENDING.value]
        
        # Sort by priority score (descending)
        pending.sort(key=lambda x: x.priority_score, reverse=True)
        
        return pending[:limit]
    
    def get_case_details(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed case information for investigator
        """
        if case_id not in self.cases:
            return None
        
        case = self.cases[case_id]
        
        return {
            'case_id': case.case_id,
            'transaction_id': case.transaction_id,
            'customer_id': case.customer_id,
            'merchant_id': case.merchant_id,
            'amount': case.amount,
            'timestamp': case.timestamp,
            'model_scores': {
                'isolation_forest': case.isolation_forest_score,
                'autoencoder': case.autoencoder_score,
                'lstm': case.lstm_score,
                'gcn': case.gcn_score,
                'ensemble': case.ensemble_score
            },
            'models_agreed': case.models_agreed,
            'risk_level': case.risk_level,
            'priority_score': case.priority_score,
            'status': case.status,
            'assigned_investigator': case.assigned_investigator,
            'created_at': case.created_at,
            'updated_at': case.updated_at,
            'decision': case.decision,
            'decision_notes': case.decision_notes
        }
    
    def get_investigator_queue(self, investigator: str) -> List[ReviewCase]:
        """
        Get all cases assigned to specific investigator
        """
        return [case for case in self.cases.values()
               if case.assigned_investigator == investigator and
               case.status in [CaseStatus.IN_REVIEW.value, CaseStatus.ESCALATED.value]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics
        """
        pending = len([c for c in self.cases.values() 
                      if c.status == CaseStatus.PENDING.value])
        in_review = len([c for c in self.cases.values() 
                        if c.status == CaseStatus.IN_REVIEW.value])
        escalated = len([c for c in self.cases.values() 
                        if c.status == CaseStatus.ESCALATED.value])
        
        return {
            'total_cases': self.statistics['total_cases'],
            'pending_cases': pending,
            'in_review_cases': in_review,
            'escalated_cases': escalated,
            'resolved_cases': self.statistics['resolved_cases'],
            'confirmed_fraud': self.statistics['confirmed_fraud'],
            'false_positives': self.statistics['false_positives'],
            'precision_percent': self.statistics['precision'],
            'backlog_days': self._estimate_backlog_days(pending)
        }
    
    def _estimate_backlog_days(self, pending_count: int) -> float:
        """
        Estimate how many days to clear backlog
        Assuming 20 cases per investigator per day
        """
        cases_per_day = 20
        return pending_count / cases_per_day if pending_count > 0 else 0
    
    def save_queue(self):
        """Persist queue to disk"""
        try:
            queue_data = {
                'cases': {
                    case_id: asdict(case) 
                    for case_id, case in self.cases.items()
                },
                'statistics': self.statistics,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2, default=str)
            
            logger.info(f"✓ Queue saved to {self.queue_file}")
            
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
    
    def load_queue(self):
        """Load queue from disk"""
        if not self.queue_file.exists():
            logger.info("No saved queue found. Starting fresh.")
            return
        
        try:
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            for case_id, case_dict in queue_data.get('cases', {}).items():
                case = ReviewCase(**case_dict)
                self.cases[case_id] = case
            
            self.statistics = queue_data.get('statistics', self.statistics)
            
            logger.info(f"✓ Loaded {len(self.cases)} cases from queue")
            
        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
    
    def export_report(self, filepath: str):
        """
        Export review queue report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'pending_cases': [
                asdict(case) for case in self.get_pending_cases(limit=100)
            ],
            'recent_resolutions': [
                asdict(case) for case in self.cases.values()
                if case.status == CaseStatus.RESOLVED.value
            ][-20:]  # Last 20
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✓ Report exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    queue = ManualReviewQueue()
    
    # Simulate adding cases
    print("\n" + "="*80)
    print("ENSEMBLE MANUAL REVIEW QUEUE")
    print("="*80 + "\n")
    
    for i in range(5):
        predictions = {
            'isolation_forest': 0.7 + i*0.05,
            'autoencoder': 0.6 + i*0.04,
            'lstm': 0.5 + i*0.03,
            'gcn': 0.8 + i*0.02,
            'ensemble_score': 0.7 + i*0.04
        }
        
        case = queue.add_case(
            transaction_id=f"TXN_{i:05d}",
            customer_id=f"CUST_{i:03d}",
            merchant_id=f"MERCH_{i:03d}",
            amount=1000 + i*500,
            predictions=predictions
        )
    
    # Get pending cases
    print("\nPENDING CASES (sorted by priority):")
    print("-" * 80)
    for case in queue.get_pending_cases():
        print(f"  {case.case_id} | Risk: {case.risk_level:8} | "
              f"Priority: {case.priority_score:6.1f} | Amount: ${case.amount:8.2f}")
    
    # Assign and resolve
    pending = queue.get_pending_cases(1)
    if pending:
        case = pending[0]
        queue.assign_case(case.case_id, "investigator_001")
        queue.resolve_case(case.case_id, FraudDecision.CONFIRMED_FRAUD.value,
                          notes="Duplicate card usage pattern detected")
    
    # Statistics
    print("\n" + "-" * 80)
    print("QUEUE STATISTICS:")
    stats = queue.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    queue.save_queue()
    print("\n" + "="*80 + "\n")
