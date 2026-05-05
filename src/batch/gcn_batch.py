"""
GCN BATCH JOB - Overnight Fraud Ring Analysis
Identifies fraud networks and connected fraudsters
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Tuple
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GCNBatch')


class GCNBatchJob:
    """
    Overnight batch processing for fraud ring detection
    Runs daily to analyze network patterns
    """
    
    def __init__(self, db_path: str = 'fraud.db', output_dir: str = 'batch_results'):
        """
        Args:
            db_path: Path to transaction database
            output_dir: Output directory for results
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.conn = None
        self.graph = None
        self.results = {}
    
    def connect_db(self):
        """Connect to transaction database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def load_transactions(self, days_back: int = 7) -> pd.DataFrame:
        """
        Load transactions from past N days
        
        Args:
            days_back: Number of days to analyze
        
        Returns:
            DataFrame of transactions
        """
        query = f"""
        SELECT customer_id, merchant_id, amount, time as timestamp, class as is_fraud
        FROM transactions
        WHERE datetime(timestamp, 'unixepoch') >= datetime('now', '-{days_back} days')
        ORDER BY timestamp DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Loaded {len(df)} transactions from past {days_back} days")
            return df
        except Exception as e:
            logger.error(f"Failed to load transactions: {e}")
            return pd.DataFrame()
    
    def build_fraud_graph(self, transactions: pd.DataFrame) -> nx.Graph:
        """
        Build bipartite customer-merchant graph
        Edges weighted by transaction amount and frequency
        
        Args:
            transactions: Transaction dataframe
        
        Returns:
            NetworkX graph
        """
        logger.info("Building fraud network graph...")
        
        graph = nx.Graph()
        
        # Get transactions flagged as fraud
        fraud_txns = transactions[transactions['is_fraud'] == 1]
        
        if len(fraud_txns) == 0:
            logger.warning("No fraud transactions found")
            return graph
        
        # Add nodes and edges
        for _, txn in fraud_txns.iterrows():
            customer = f"C_{txn['customer_id']}"
            merchant = f"M_{txn['merchant_id']}"
            
            graph.add_node(customer, node_type='customer')
            graph.add_node(merchant, node_type='merchant')
            
            # Edge weight: transaction amount
            if graph.has_edge(customer, merchant):
                graph[customer][merchant]['weight'] += txn['amount']
                graph[customer][merchant]['count'] += 1
            else:
                graph.add_edge(
                    customer, merchant,
                    weight=txn['amount'],
                    count=1,
                    amount_sum=txn['amount']
                )
        
        logger.info(f"Graph created: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        self.graph = graph
        return graph
    
    def detect_fraud_rings(self) -> List[Dict[str, Any]]:
        """
        Detect fraud rings using graph community detection
        
        Returns:
            List of detected fraud rings with members and stats
        """
        if self.graph is None or len(self.graph) == 0:
            logger.warning("Graph is empty")
            return []
        
        logger.info("Detecting fraud rings using community detection...")
        
        rings = []
        
        # Find connected components (communities)
        for component in nx.connected_components(self.graph):
            if len(component) < 2:
                continue
            
            subgraph = self.graph.subgraph(component)
            
            # Get ring statistics
            customers = [n for n in component if n.startswith('C_')]
            merchants = [n for n in component if n.startswith('M_')]
            
            # Calculate metrics
            total_amount = sum(
                data.get('amount_sum', 0) 
                for _, _, data in subgraph.edges(data=True)
            )
            
            edge_count = len(subgraph.edges())
            density = nx.density(subgraph)
            
            # Identify ring members by degree
            degree_dict = dict(subgraph.degree())
            top_members = sorted(
                degree_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            ring_info = {
                'ring_id': f"RING_{len(rings):04d}",
                'size': len(component),
                'customer_count': len(customers),
                'merchant_count': len(merchants),
                'total_transaction_amount': float(total_amount),
                'edge_count': edge_count,
                'network_density': float(density),
                'members': customers,
                'merchants': merchants,
                'key_members': [m[0] for m in top_members],
                'risk_score': float(min(density * total_amount / 1000, 1.0))
            }
            
            rings.append(ring_info)
        
        rings = sorted(rings, key=lambda x: x['risk_score'], reverse=True)
        logger.info(f"Detected {len(rings)} fraud rings")
        
        return rings
    
    def calculate_customer_risk_scores(
        self, 
        transactions: pd.DataFrame,
        fraud_rings: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate risk scores for all customers
        Based on fraud history, network connections, and transaction patterns
        
        Args:
            transactions: All transactions
            fraud_rings: Detected fraud rings
        
        Returns:
            DataFrame with customer risk scores
        """
        logger.info("Calculating customer risk scores...")
        
        # Get unique customers
        customers = transactions['customer_id'].unique()
        
        risk_data = []
        
        for customer_id in customers:
            customer_txns = transactions[transactions['customer_id'] == customer_id]
            
            # Basic metrics
            fraud_count = (customer_txns['is_fraud'] == 1).sum()
            total_txns = len(customer_txns)
            fraud_rate = fraud_count / max(total_txns, 1)
            
            # Network risk: in fraud ring?
            ring_risk = 0.0
            in_rings = []
            for ring in fraud_rings:
                if f"C_{customer_id}" in ring['members']:
                    ring_risk = max(ring_risk, ring['risk_score'])
                    in_rings.append(ring['ring_id'])
            
            # Transaction pattern risk
            amounts = customer_txns['amount'].values
            amount_std = np.std(amounts) if len(amounts) > 1 else 0
            amount_mean = np.mean(amounts)
            amount_zscore = (amount_std / max(amount_mean, 1)) if amount_mean > 0 else 0
            
            # Composite risk
            fraud_risk = fraud_rate * 0.4
            network_risk = ring_risk * 0.4
            pattern_risk = min(amount_zscore / 2, 1.0) * 0.2
            
            total_risk = fraud_risk + network_risk + pattern_risk
            
            risk_data.append({
                'customer_id': customer_id,
                'fraud_count': fraud_count,
                'total_transactions': total_txns,
                'fraud_rate': float(fraud_rate),
                'fraud_risk_component': float(fraud_risk),
                'network_risk_component': float(network_risk),
                'pattern_risk_component': float(pattern_risk),
                'total_risk_score': float(min(total_risk, 1.0)),
                'in_fraud_rings': in_rings,
                'transaction_amount_std': float(amount_std),
                'transaction_amount_mean': float(amount_mean)
            })
        
        risk_df = pd.DataFrame(risk_data)
        risk_df = risk_df.sort_values('total_risk_score', ascending=False)
        
        logger.info(f"Calculated risk scores for {len(risk_df)} customers")
        return risk_df
    
    def generate_report(self, fraud_rings: List[Dict], risk_scores: pd.DataFrame) -> Dict:
        """
        Generate comprehensive fraud analysis report
        
        Args:
            fraud_rings: Detected fraud rings
            risk_scores: Customer risk scores
        
        Returns:
            Report dictionary
        """
        logger.info("Generating report...")
        
        # High-risk customers (for immediate action)
        high_risk = risk_scores[risk_scores['total_risk_score'] > 0.7]
        
        # Priority fraud rings (by total risk)
        top_rings = sorted(
            fraud_rings,
            key=lambda x: x['risk_score'],
            reverse=True
        )[:10]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_fraud_rings': len(fraud_rings),
                'total_customers_analyzed': len(risk_scores),
                'high_risk_customers': len(high_risk),
                'customers_in_rings': sum(1 for _, row in risk_scores.iterrows() if row['in_fraud_rings'])
            },
            'top_fraud_rings': top_rings,
            'high_risk_customers': high_risk[['customer_id', 'total_risk_score', 'in_fraud_rings']].to_dict('records'),
            'recommendations': self._generate_recommendations(fraud_rings, risk_scores)
        }
        
        return report
    
    def _generate_recommendations(self, fraud_rings: List[Dict], risk_scores: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if len(fraud_rings) > 5:
            recommendations.append(
                f"⚠️  High number of fraud rings detected ({len(fraud_rings)}). "
                "Consider increased monitoring."
            )
        
        high_risk = risk_scores[risk_scores['total_risk_score'] > 0.7]
        if len(high_risk) > 0:
            recommendations.append(
                f"🚨 {len(high_risk)} high-risk customers identified. "
                "Recommend immediate review and possible account suspension."
            )
        
        # Dense networks
        dense_rings = [r for r in fraud_rings if r['network_density'] > 0.5]
        if dense_rings:
            recommendations.append(
                f"📊 {len(dense_rings)} highly connected fraud rings detected. "
                "Investigate merchant relationships."
            )
        
        if not recommendations:
            recommendations.append("✓ Fraud pattern analysis shows normal activity.")
        
        return recommendations
    
    def save_results(self):
        """Save batch job results to files"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save report
        report_path = f"{self.output_dir}/fraud_rings_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_path
    
    def run_batch(self, days_back: int = 7):
        """
        Execute complete batch job
        
        Args:
            days_back: Number of days to analyze
        
        Returns:
            Batch results
        """
        logger.info("=" * 80)
        logger.info("GCN BATCH JOB - OVERNIGHT FRAUD RING ANALYSIS")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Connect and load data
        if not self.connect_db():
            logger.error("Failed to connect to database")
            return None
        
        transactions = self.load_transactions(days_back)
        if len(transactions) == 0:
            logger.error("No transactions loaded")
            return None
        
        # Step 2: Build graph
        self.build_fraud_graph(transactions)
        
        # Step 3: Detect fraud rings
        fraud_rings = self.detect_fraud_rings()
        
        # Step 4: Calculate risk scores
        risk_scores = self.calculate_customer_risk_scores(transactions, fraud_rings)
        
        # Step 5: Generate report
        report = self.generate_report(fraud_rings, risk_scores)
        self.results = report
        
        # Step 6: Save results
        self.save_results()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info(f"Batch job completed in {elapsed:.2f}s")
        logger.info(f"Fraud rings detected: {len(fraud_rings)}")
        logger.info(f"High-risk customers: {len(risk_scores[risk_scores['total_risk_score'] > 0.7])}")
        logger.info("=" * 80)
        
        return report
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Scheduler for automated batch jobs
def schedule_batch_job(time_str: str = "02:00"):
    """
    Schedule GCN batch job to run at specific time
    
    Example: schedule_batch_job("02:00")  # 2 AM daily
    """
    try:
        import schedule
        import time as time_module
        
        def batch_job():
            job = GCNBatchJob()
            job.run_batch(days_back=7)
            job.close()
        
        schedule.every().day.at(time_str).do(batch_job)
        logger.info(f"Batch job scheduled for {time_str} daily")
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time_module.sleep(60)
    
    except ImportError:
        logger.error("schedule module not installed. Install with: pip install schedule")


if __name__ == "__main__":
    # Example: Run batch job manually
    job = GCNBatchJob()
    results = job.run_batch(days_back=7)
    job.close()
    
    if results:
        print("\n📊 BATCH RESULTS:")
        print(json.dumps(results['summary'], indent=2))
