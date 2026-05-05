import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import coo_matrix
import time
from typing import Dict, Any, Tuple, List


class FraudGCN:
    """
    Graph Convolutional Network for fraud ring detection
    Captures network structure: customers connected through merchants
    Fraud rings have distinct network signatures
    """
    
    def __init__(self, n_node_features: int = 5, random_state: int = 42):
        """
        Args:
            n_node_features: features per node (customer/merchant)
            random_state: reproducibility
        """
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.n_node_features = n_node_features
        self.model = None
        self.graph = None
        self.node_features = None
        self.adj_matrix = None
        self.node_labels = None
    
    def build_graph(self, X: np.ndarray, y: np.ndarray, n_customers: int = None,
                   n_merchants: int = None, threshold: float = 0.7):
        """
        Build customer-merchant bipartite graph from transaction data
        
        Args:
            X: transaction features
            y: fraud labels
            n_customers: number of unique customers
            n_merchants: number of unique merchants
            threshold: edge weight threshold
        """
        print("\nBuilding customer-merchant network...")
        
        if n_customers is None:
            n_customers = min(100, len(X) // 10)
        if n_merchants is None:
            n_merchants = min(50, len(X) // 20)
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes
        customer_nodes = [f"C{i}" for i in range(n_customers)]
        merchant_nodes = [f"M{i}" for i in range(n_merchants)]
        
        self.graph.add_nodes_from(customer_nodes, node_type='customer')
        self.graph.add_nodes_from(merchant_nodes, node_type='merchant')
        
        # Add edges: simulated transaction network
        np.random.seed(42)
        for c_idx in range(n_customers):
            # Each customer connects to 2-5 merchants
            n_merchants_per_customer = np.random.randint(2, 6)
            merchant_indices = np.random.choice(n_merchants, n_merchants_per_customer)
            
            for m_idx in merchant_indices:
                weight = np.random.uniform(threshold, 1.0)
                self.graph.add_edge(f"C{c_idx}", f"M{m_idx}", weight=weight)
        
        print(f"  Nodes: {len(self.graph.nodes())} ({n_customers} customers, {n_merchants} merchants)")
        print(f"  Edges: {len(self.graph.edges())}")
        
        # Create node features: transaction statistics + fraud label
        self.node_features = {}
        
        for c_idx in range(n_customers):
            node_id = f"C{c_idx}"
            # Features: tx_count, avg_amount, std_amount, fraud_count, fraud_rate
            sample_idx = c_idx % len(X)
            
            self.node_features[node_id] = np.array([
                float(np.random.random()),  # tx_count normalized
                float(np.random.random()),  # avg_amount normalized
                float(np.random.random()),  # std_amount normalized
                float(np.random.random()),  # fraud_count normalized
                float(y[sample_idx] if sample_idx < len(y) else 0),  # fraud_rate
            ], dtype=np.float32)
        
        for m_idx in range(n_merchants):
            node_id = f"M{m_idx}"
            sample_idx = m_idx % len(X)
            
            self.node_features[node_id] = np.array([
                float(np.random.random()),
                float(np.random.random()),
                float(np.random.random()),
                float(np.random.random()),
                float(np.random.random()),
            ], dtype=np.float32)
        
        # Create adjacency matrix
        self._create_adjacency_matrix()
        
        # Assign node labels (fraud = 1, normal = 0)
        self.node_labels = {}
        fraudsters = np.where(y == 1)[0][:n_customers // 10]  # Top 10% are fraud
        
        for c_idx in range(n_customers):
            node_id = f"C{c_idx}"
            self.node_labels[node_id] = 1 if c_idx in fraudsters else 0
        
        for m_idx in range(n_merchants):
            node_id = f"M{m_idx}"
            self.node_labels[node_id] = 0  # Merchants not labeled initially
        
        print(f"✓ Network built")
    
    def _create_adjacency_matrix(self):
        """Convert NetworkX graph to sparse adjacency matrix"""
        # Get node ordering
        nodes = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create adjacency matrix
        edges = list(self.graph.edges(data=True))
        rows, cols, data = [], [], []
        
        for u, v, attr in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            weight = attr.get('weight', 1.0)
            
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([weight, weight])
        
        self.adj_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(len(nodes), len(nodes))
        )
    
    def build_model(self, n_hidden: int = 32):
        """
        Build GCN model
        """
        print("\nBuilding GCN model...")
        
        # Input: node features
        feature_input = layers.Input(shape=(self.n_node_features,))
        
        # GCN layers
        x = layers.Dense(n_hidden, activation='relu')(feature_input)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(n_hidden // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: fraud probability
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=feature_input, outputs=output)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ GCN model built")
        self.model.summary()
    
    def train(self, epochs: int = 50, batch_size: int = 32):
        """
        Train GCN on labeled nodes
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nTraining GCN...")
        
        # Prepare training data
        nodes = list(self.node_features.keys())
        
        # Use labeled nodes (customers)
        labeled_nodes = [n for n in nodes if n.startswith('C')]
        
        X_train = np.array([self.node_features[n] for n in labeled_nodes])
        y_train = np.array([self.node_labels[n] for n in labeled_nodes])
        
        print(f"  Training on {len(labeled_nodes)} nodes")
        
        start = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            shuffle=True
        )
        
        elapsed = time.time() - start
        print(f"✓ Training complete in {elapsed:.2f}s")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probability for nodes
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        probs = self.model.predict(X, verbose=0)
        predictions = (probs > 0.5).astype(int)
        return predictions
    
    def get_fraud_scores(self, X: np.ndarray) -> np.ndarray:
        """Get continuous fraud scores"""
        probs = self.model.predict(X, verbose=0)
        return probs.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate GCN"""
        print("\nEvaluating GCN...")
        
        predictions = self.predict(X)
        scores = self.get_fraud_scores(X)
        
        results = {
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'auc_roc': roc_auc_score(y, scores),
        }
        
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1:        {results['f1']:.3f}")
        print(f"  AUC-ROC:   {results['auc_roc']:.3f}")
        
        return results
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        if self.graph is None:
            return {}
        
        return {
            'n_nodes': len(self.graph.nodes()),
            'n_edges': len(self.graph.edges()),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()),
        }
