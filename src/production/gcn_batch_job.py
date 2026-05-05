"""
GCN Batch Job: Overnight Fraud Ring Analysis
Runs nightly to detect fraud rings and connected fraudsters
"""

import numpy as np
import networkx as nx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCNBatchJob:
    """
    Overnight GCN batch processing for fraud ring detection
    Analyzes network structure to identify coordinated fraud
    """
    
    def __init__(self, output_dir: str = 'batch_results'):
        """
        Args:
            output_dir: Directory for batch results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.graph = None
        self.fraud_rings = []
        self.flagged_nodes = {}
        self.execution_log = {
            'start_time': None,
            'end_time': None,
            'status': 'PENDING',
            'fraud_rings_detected': 0,
            'suspicious_nodes': 0,
            'duration_seconds': 0
        }
    
    def load_transaction_data(self, transactions: List[Dict[str, Any]]):
        """
        Load transaction data and build graph
        
        Args:
            transactions: List of transaction records with customer_id, merchant_id, amount, class
        """
        logger.info(f"Loading {len(transactions)} transactions...")
        
        # Build bipartite graph (customers + merchants)
        self.graph = nx.Graph()
        
        # Track customer and merchant interactions
        customer_merchants = {}
        merchant_customers = {}
        fraud_customers = set()
        
        for txn in transactions:
            customer_id = f"C{txn.get('customer_id', 0)}"
            merchant_id = f"M{txn.get('merchant_id', 0)}"
            amount = txn.get('amount', 0)
            is_fraud = txn.get('class', 0) == 1
            
            # Add nodes
            self.graph.add_node(customer_id, node_type='customer')
            self.graph.add_node(merchant_id, node_type='merchant')
            
            # Add edge (weight = transaction amount)
            if self.graph.has_edge(customer_id, merchant_id):
                self.graph[customer_id][merchant_id]['weight'] += amount
                self.graph[customer_id][merchant_id]['count'] += 1
            else:
                self.graph.add_edge(customer_id, merchant_id, weight=amount, count=1)
            
            # Track fraud
            if is_fraud:
                fraud_customers.add(customer_id)
            
            # Track relationships
            if customer_id not in customer_merchants:
                customer_merchants[customer_id] = set()
            customer_merchants[customer_id].add(merchant_id)
            
            if merchant_id not in merchant_customers:
                merchant_customers[merchant_id] = set()
            merchant_customers[merchant_id].add(customer_id)
        
        logger.info(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        logger.info(f"✓ Fraud customers: {len(fraud_customers)}")
        
        return fraud_customers, customer_merchants, merchant_customers
    
    def detect_fraud_rings(self, fraud_customers: set, 
                          customer_merchants: Dict) -> List[set]:
        """
        Detect fraud rings via network analysis
        
        A fraud ring = group of customers using same merchants
        """
        logger.info("Detecting fraud rings...")
        
        rings = []
        visited = set()
        
        for customer in fraud_customers:
            if customer in visited:
                continue
            
            # Find all fraud customers connected through merchants
            ring = {customer}
            to_visit = {customer}
            visited.add(customer)
            
            while to_visit:
                current = to_visit.pop()
                
                # Find customers who share merchants with current
                shared_merchants = customer_merchants.get(current, set())
                
                for merchant in shared_merchants:
                    for other_customer in customer_merchants:
                        if (other_customer not in ring and 
                            other_customer in fraud_customers and
                            merchant in customer_merchants.get(other_customer, set())):
                            
                            ring.add(other_customer)
                            visited.add(other_customer)
                            to_visit.add(other_customer)
            
            if len(ring) > 1:  # Ring = 2+ customers
                rings.append(ring)
        
        logger.info(f"✓ Detected {len(rings)} fraud rings")
        self.fraud_rings = rings
        
        return rings
    
    def analyze_ring_characteristics(self, rings: List[set]) -> List[Dict[str, Any]]:
        """
        Analyze characteristics of each fraud ring
        """
        logger.info("Analyzing ring characteristics...")
        
        ring_analysis = []
        
        for ring_id, ring in enumerate(rings):
            # Get nodes and edges in ring
            ring_nodes = list(ring)
            ring_subgraph = self.graph.subgraph(ring_nodes)
            
            # Calculate metrics
            total_transactions = sum(
                data.get('count', 0) 
                for _, _, data in ring_subgraph.edges(data=True)
            )
            
            total_amount = sum(
                data.get('weight', 0) 
                for _, _, data in ring_subgraph.edges(data=True)
            )
            
            avg_degree = np.mean([d for n, d in ring_subgraph.degree()])
            
            # Identify merchant hubs
            merchant_nodes = [n for n in ring_nodes if n.startswith('M')]
            customer_nodes = [n for n in ring_nodes if n.startswith('C')]
            
            analysis = {
                'ring_id': f'RING_{ring_id:05d}',
                'size': len(ring),
                'customers': len(customer_nodes),
                'merchants': len(merchant_nodes),
                'member_ids': list(ring),
                'total_transactions': total_transactions,
                'total_amount': float(total_amount),
                'avg_degree': float(avg_degree),
                'risk_score': self._calculate_ring_risk_score(
                    len(ring), total_transactions, total_amount
                ),
                'detected_at': datetime.now().isoformat()
            }
            
            ring_analysis.append(analysis)
        
        # Sort by risk score
        ring_analysis.sort(key=lambda x: x['risk_score'], reverse=True)
        
        logger.info(f"✓ Analyzed {len(ring_analysis)} rings")
        
        return ring_analysis
    
    def _calculate_ring_risk_score(self, size: int, tx_count: int, 
                                   amount: float) -> float:
        """Calculate fraud ring risk score (0-100)"""
        # Larger rings = higher risk
        size_score = min(size * 10, 40)
        
        # More transactions = higher risk
        tx_score = min(tx_count * 0.5, 30)
        
        # Larger amounts = higher risk
        amount_score = min(amount / 10000, 30)
        
        return float(size_score + tx_score + amount_score)
    
    def flag_suspicious_nodes(self, ring_analysis: List[Dict[str, Any]]):
        """
        Flag customers and merchants involved in fraud rings
        """
        logger.info("Flagging suspicious nodes...")
        
        for ring in ring_analysis:
            for node_id in ring['member_ids']:
                if node_id not in self.flagged_nodes:
                    self.flagged_nodes[node_id] = {
                        'rings': [],
                        'risk_score': 0.0,
                        'flagged_at': datetime.now().isoformat()
                    }
                
                self.flagged_nodes[node_id]['rings'].append(ring['ring_id'])
                self.flagged_nodes[node_id]['risk_score'] = max(
                    self.flagged_nodes[node_id]['risk_score'],
                    ring['risk_score']
                )
        
        logger.info(f"✓ Flagged {len(self.flagged_nodes)} suspicious nodes")
    
    def generate_alerts(self, ring_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate actionable alerts
        """
        logger.info("Generating alerts...")
        
        alerts = []
        
        for ring in ring_analysis[:10]:  # Top 10 rings
            alert = {
                'alert_id': f'ALERT_{ring["ring_id"]}',
                'type': 'FRAUD_RING',
                'severity': self._get_severity(ring['risk_score']),
                'ring_id': ring['ring_id'],
                'members_count': ring['size'],
                'customers': ring['customers'],
                'merchants': ring['merchants'],
                'total_transactions': ring['total_transactions'],
                'total_amount': ring['total_amount'],
                'risk_score': ring['risk_score'],
                'recommended_action': self._get_recommended_action(ring['risk_score']),
                'generated_at': datetime.now().isoformat()
            }
            alerts.append(alert)
        
        logger.info(f"✓ Generated {len(alerts)} alerts")
        
        return alerts
    
    def _get_severity(self, risk_score: float) -> str:
        """Map risk score to severity level"""
        if risk_score > 80:
            return "CRITICAL"
        elif risk_score > 60:
            return "HIGH"
        elif risk_score > 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommended_action(self, risk_score: float) -> str:
        """Get recommended action based on risk"""
        if risk_score > 80:
            return "Immediate investigation + Block all members"
        elif risk_score > 60:
            return "Investigation + Enhanced monitoring"
        elif risk_score > 40:
            return "Enhanced monitoring"
        else:
            return "Flag for review"
    
    def save_results(self, ring_analysis: List[Dict[str, Any]], 
                    alerts: List[Dict[str, Any]]):
        """Save batch job results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ring analysis
        ring_file = self.output_dir / f"fraud_rings_{timestamp}.json"
        with open(ring_file, 'w') as f:
            json.dump(ring_analysis, f, indent=2)
        logger.info(f"✓ Ring analysis saved to {ring_file}")
        
        # Save alerts
        alert_file = self.output_dir / f"alerts_{timestamp}.json"
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        logger.info(f"✓ Alerts saved to {alert_file}")
        
        # Save flagged nodes
        nodes_file = self.output_dir / f"flagged_nodes_{timestamp}.json"
        with open(nodes_file, 'w') as f:
            json.dump(self.flagged_nodes, f, indent=2)
        logger.info(f"✓ Flagged nodes saved to {nodes_file}")
    
    def run(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute complete batch job
        """
        self.execution_log['start_time'] = datetime.now().isoformat()
        logger.info("="*80)
        logger.info("GCN BATCH JOB: OVERNIGHT FRAUD RING ANALYSIS")
        logger.info("="*80)
        
        try:
            # Step 1: Load and build graph
            fraud_customers, customer_merchants, merchant_customers = \
                self.load_transaction_data(transactions)
            
            # Step 2: Detect rings
            rings = self.detect_fraud_rings(fraud_customers, customer_merchants)
            
            # Step 3: Analyze rings
            ring_analysis = self.analyze_ring_characteristics(rings)
            
            # Step 4: Flag nodes
            self.flag_suspicious_nodes(ring_analysis)
            
            # Step 5: Generate alerts
            alerts = self.generate_alerts(ring_analysis)
            
            # Step 6: Save results
            self.save_results(ring_analysis, alerts)
            
            # Update execution log
            self.execution_log['end_time'] = datetime.now().isoformat()
            self.execution_log['status'] = 'SUCCESS'
            self.execution_log['fraud_rings_detected'] = len(rings)
            self.execution_log['suspicious_nodes'] = len(self.flagged_nodes)
            
            start = datetime.fromisoformat(self.execution_log['start_time'])
            end = datetime.fromisoformat(self.execution_log['end_time'])
            self.execution_log['duration_seconds'] = (end - start).total_seconds()
            
            logger.info("="*80)
            logger.info(f"JOB COMPLETE: {len(rings)} rings, {len(alerts)} alerts")
            logger.info(f"Duration: {self.execution_log['duration_seconds']:.2f} seconds")
            logger.info("="*80)
            
            return {
                'status': 'SUCCESS',
                'fraud_rings': ring_analysis,
                'alerts': alerts,
                'flagged_nodes': self.flagged_nodes,
                'execution_log': self.execution_log
            }
            
        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            self.execution_log['status'] = 'FAILED'
            self.execution_log['end_time'] = datetime.now().isoformat()
            
            return {
                'status': 'FAILED',
                'error': str(e),
                'execution_log': self.execution_log
            }


if __name__ == "__main__":
    # Example: Create sample transactions
    from data_prep import create_sample_dataset
    
    X, y = create_sample_dataset(n_samples=1000)
    
    transactions = []
    for i in range(len(X)):
        transactions.append({
            'customer_id': i % 100,
            'merchant_id': (i // 10) % 50,
            'amount': X[i][0] * 100,
            'class': int(y[i])
        })
    
    # Run batch job
    batch_job = GCNBatchJob(output_dir='fraud-detection-system/batch_results')
    result = batch_job.run(transactions)
    
    if result['status'] == 'SUCCESS':
        print(f"\n✓ Detected {result['execution_log']['fraud_rings_detected']} fraud rings")
        print(f"✓ Flagged {result['execution_log']['suspicious_nodes']} suspicious nodes")
        print(f"✓ Generated {len(result['alerts'])} alerts")
