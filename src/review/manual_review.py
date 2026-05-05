"""
ENSEMBLE MANUAL REVIEW QUEUE
High-stakes fraud review system with multiple model consensus
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import sqlite3
import logging
from collections import deque
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ManualReviewQueue')


class ReviewStatus(str, Enum):
    """Review queue item status"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class ReviewPriority(str, Enum):
    """Item priority level"""
    CRITICAL = "critical"  # Multiple models agree, high amount
    HIGH = "high"          # Strong consensus
    MEDIUM = "medium"      # Moderate agreement
    LOW = "low"            # Single model flagged


class ManualReviewQueue:
    """
    Queue for high-stakes fraud decisions requiring manual review
    Integrates ensemble predictions with reviewer workflow
    """
    
    def __init__(self, db_path: str = 'review_queue.db', max_queue_size: int = 10000):
        """
        Args:
            db_path: Database for persistent storage
            max_queue_size: Maximum queue length
        """
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.queue = deque(maxlen=max_queue_size)
        self.lock = threading.Lock()
        
        # Initialize database
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_items (
                    id TEXT PRIMARY KEY,
                    transaction_id TEXT UNIQUE,
                    customer_id TEXT,
                    merchant_id TEXT,
                    amount REAL,
                    timestamp TEXT,
                    status TEXT,
                    priority TEXT,
                    created_at TEXT,
                    reviewed_at TEXT,
                    reviewer_id TEXT,
                    reviewer_decision TEXT,
                    reviewer_notes TEXT,
                    model_consensus REAL,
                    models_agree INTEGER,
                    model_predictions TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    total_reviewed INTEGER,
                    approved INTEGER,
                    rejected INTEGER,
                    escalated INTEGER,
                    avg_review_time_minutes REAL,
                    accuracy_vs_models REAL,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database init failed: {e}")
    
    def add_to_queue(
        self,
        transaction_id: str,
        customer_id: str,
        merchant_id: str,
        amount: float,
        model_predictions: Dict[str, Any],
        timestamp: str = None
    ) -> str:
        """
        Add transaction to manual review queue
        
        Args:
            transaction_id: Unique transaction ID
            customer_id: Customer identifier
            merchant_id: Merchant identifier
            amount: Transaction amount
            model_predictions: Predictions from all models
            timestamp: Transaction timestamp
        
        Returns:
            Review item ID
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        # Calculate consensus metrics
        model_scores = [
            v.get('fraud_probability', v.get('is_fraud', 0))
            for v in model_predictions.values()
            if isinstance(v, dict)
        ]
        
        models_agree = sum(1 for s in model_scores if s > 0.5)
        avg_score = sum(model_scores) / max(len(model_scores), 1) if model_scores else 0
        
        # Determine priority
        priority = self._calculate_priority(
            models_agree,
            avg_score,
            amount,
            len(model_scores)
        )
        
        # Create review item
        review_id = f"REV_{transaction_id}_{int(datetime.now().timestamp())}"
        
        item = {
            'review_id': review_id,
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'timestamp': timestamp,
            'status': ReviewStatus.PENDING.value,
            'priority': priority.value,
            'created_at': datetime.now().isoformat(),
            'models_agree': models_agree,
            'model_consensus': float(avg_score),
            'model_predictions': model_predictions,
            'reviewed_at': None,
            'reviewer_id': None,
            'reviewer_decision': None,
            'reviewer_notes': None
        }
        
        # Add to in-memory queue
        with self.lock:
            self.queue.append(item)
        
        # Persist to database
        self._save_item(item)
        
        logger.info(
            f"Added to review queue: {review_id} | "
            f"Priority: {priority.value} | "
            f"Consensus: {avg_score:.2f}"
        )
        
        return review_id
    
    def _calculate_priority(
        self,
        models_agree: int,
        avg_score: float,
        amount: float,
        total_models: int
    ) -> ReviewPriority:
        """Calculate priority based on consensus and risk factors"""
        
        # Strong consensus (3+ out of 5+ models agree)
        if models_agree >= 3 and total_models >= 5:
            if avg_score > 0.8 or amount > 5000:
                return ReviewPriority.CRITICAL
            return ReviewPriority.HIGH
        
        # Moderate consensus
        if models_agree >= 2:
            if amount > 10000:
                return ReviewPriority.CRITICAL
            return ReviewPriority.HIGH
        
        # Weak consensus
        if avg_score > 0.7:
            return ReviewPriority.MEDIUM
        
        return ReviewPriority.LOW
    
    def get_pending_items(
        self,
        priority: Optional[ReviewPriority] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get pending review items
        
        Args:
            priority: Filter by priority (None = all)
            limit: Maximum items to return
        
        Returns:
            List of pending items sorted by priority
        """
        with self.lock:
            items = [
                item for item in self.queue
                if item['status'] == ReviewStatus.PENDING.value
            ]
        
        if priority:
            items = [i for i in items if i['priority'] == priority.value]
        
        # Sort by priority level (critical > high > medium > low)
        priority_order = {
            ReviewPriority.CRITICAL.value: 0,
            ReviewPriority.HIGH.value: 1,
            ReviewPriority.MEDIUM.value: 2,
            ReviewPriority.LOW.value: 3
        }
        items.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return items[:limit]
    
    def submit_review(
        self,
        review_id: str,
        reviewer_id: str,
        decision: str,
        notes: str = ""
    ) -> bool:
        """
        Submit manual review decision
        
        Args:
            review_id: Review item ID
            reviewer_id: Reviewer identifier
            decision: "approved" or "rejected"
            notes: Reviewer notes
        
        Returns:
            Success status
        """
        if decision not in ['approved', 'rejected', 'escalated']:
            logger.error(f"Invalid decision: {decision}")
            return False
        
        try:
            with self.lock:
                item = next(
                    (i for i in self.queue if i['review_id'] == review_id),
                    None
                )
                
                if not item:
                    logger.error(f"Review item not found: {review_id}")
                    return False
                
                item['status'] = decision
                item['reviewer_id'] = reviewer_id
                item['reviewer_decision'] = decision
                item['reviewer_notes'] = notes
                item['reviewed_at'] = datetime.now().isoformat()
            
            # Update database
            self._update_item(item)
            
            logger.info(
                f"Review submitted: {review_id} | "
                f"Decision: {decision} | "
                f"Reviewer: {reviewer_id}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to submit review: {e}")
            return False
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get review queue statistics
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Statistics dictionary
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.lock:
            items = list(self.queue)
        
        # Filter by date
        recent_items = [
            i for i in items
            if i['created_at'] >= cutoff_date
        ]
        
        # Count by status
        status_counts = {}
        for status in ReviewStatus:
            count = sum(1 for i in recent_items if i['status'] == status.value)
            status_counts[status.value] = count
        
        # Count by priority
        priority_counts = {}
        for priority in ReviewPriority:
            count = sum(1 for i in recent_items if i['priority'] == priority.value)
            priority_counts[priority.value] = count
        
        # Review accuracy (comparing with models)
        reviewed_items = [
            i for i in recent_items
            if i['reviewer_decision'] is not None
        ]
        
        accuracy = 0.0
        if reviewed_items:
            matches = sum(
                1 for i in reviewed_items
                if (i['reviewer_decision'] == 'approved' and i['model_consensus'] > 0.5) or
                   (i['reviewer_decision'] == 'rejected' and i['model_consensus'] <= 0.5)
            )
            accuracy = matches / len(reviewed_items)
        
        # Response time
        review_times = [
            (
                datetime.fromisoformat(i['reviewed_at']) -
                datetime.fromisoformat(i['created_at'])
            ).total_seconds() / 60
            for i in reviewed_items
            if i['reviewed_at']
        ]
        
        avg_review_time = (
            sum(review_times) / len(review_times)
            if review_times else 0
        )
        
        return {
            'period_days': days,
            'total_items': len(recent_items),
            'status_breakdown': status_counts,
            'priority_breakdown': priority_counts,
            'reviewed_items': len(reviewed_items),
            'review_accuracy_vs_models': float(accuracy),
            'avg_review_time_minutes': float(avg_review_time),
            'high_priority_pending': sum(
                1 for i in recent_items
                if i['status'] == ReviewStatus.PENDING.value and
                   i['priority'] in [ReviewPriority.CRITICAL.value, ReviewPriority.HIGH.value]
            )
        }
    
    def _save_item(self, item: Dict[str, Any]):
        """Save review item to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO review_items
                (id, transaction_id, customer_id, merchant_id, amount, timestamp,
                 status, priority, created_at, models_agree, model_consensus, model_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['review_id'],
                item['transaction_id'],
                item['customer_id'],
                item['merchant_id'],
                item['amount'],
                item['timestamp'],
                item['status'],
                item['priority'],
                item['created_at'],
                item['models_agree'],
                item['model_consensus'],
                json.dumps(item['model_predictions'], default=str)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save item: {e}")
    
    def _update_item(self, item: Dict[str, Any]):
        """Update review item in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE review_items
                SET status = ?, reviewed_at = ?, reviewer_id = ?,
                    reviewer_decision = ?, reviewer_notes = ?
                WHERE id = ?
            ''', (
                item['status'],
                item['reviewed_at'],
                item['reviewer_id'],
                item['reviewer_decision'],
                item['reviewer_notes'],
                item['review_id']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update item: {e}")
    
    def export_for_analytics(self, output_path: str = "review_queue_export.json"):
        """Export queue data for analytics"""
        with self.lock:
            items = list(self.queue)
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_items': len(items),
            'items': items
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Queue exported to {output_path}")


# Example usage
if __name__ == "__main__":
    queue = ManualReviewQueue()
    
    # Example: Add item to queue
    model_preds = {
        'isolation_forest': {'fraud_probability': 0.9, 'is_fraud': True},
        'autoencoder': {'fraud_probability': 0.85, 'is_fraud': True},
        'lstm': {'fraud_probability': 0.7, 'is_fraud': True},
        'gcn': {'fraud_probability': 0.95, 'is_fraud': True}
    }
    
    review_id = queue.add_to_queue(
        transaction_id="TXN_123456",
        customer_id="CUST_789",
        merchant_id="MERCH_456",
        amount=2500.00,
        model_predictions=model_preds
    )
    
    # Get pending items
    pending = queue.get_pending_items(limit=5)
    print(f"\nPending reviews: {len(pending)}")
    for item in pending:
        print(f"  {item['review_id']}: {item['priority']}")
    
    # Get statistics
    stats = queue.get_statistics(days=7)
    print(f"\n📊 Queue Statistics:")
    print(json.dumps(stats, indent=2))
