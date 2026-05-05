import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from typing import Dict, Any, List, Tuple
import time


class ModelEvaluator:
    """Comprehensive evaluation and comparison of fraud detection models"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
        self.timings = {}
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray, y_score: np.ndarray = None,
                      latency_ms: float = None) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model_name: name of the model
            y_true: true labels
            y_pred: binary predictions (0/1)
            y_score: probability scores for ROC/PR curves
            latency_ms: prediction latency in milliseconds
        
        Returns:
            metrics dictionary
        """
        if y_score is None:
            y_score = y_pred
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_score),
        }
        
        # Add latency if provided
        if latency_ms is not None:
            metrics['latency_ms'] = latency_ms
        
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_score': y_score,
            'y_true': y_true
        }
        
        return metrics
    
    def print_comparison_table(self):
        """Print formatted comparison table of all models"""
        if not self.results:
            print("No results to display. Run evaluate_model() first.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(df.round(4).to_string())
        print("=" * 100)
        
        return df
    
    def get_best_model(self, metric: str = 'f1') -> str:
        """Get model with best performance on given metric"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), 
                        key=lambda x: x[1].get(metric, 0))[0]
        return best_model
    
    def plot_roc_curves(self, filepath: str = None, figsize: Tuple = (14, 10)):
        """
        Plot ROC curves for all models
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            y_true = preds['y_true']
            y_score = preds['y_score']
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'{model_name} ROC Curve', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {filepath}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, filepath: str = None, figsize: Tuple = (14, 10)):
        """
        Plot precision-recall curves for all models
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            y_true = preds['y_true']
            y_score = preds['y_score']
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auc_pr = roc_auc_score(y_true, y_score)
            
            ax.plot(recall, precision, linewidth=2, label=f'AUC = {auc_pr:.3f}')
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(f'{model_name} PR Curve', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"PR curves saved to {filepath}")
        
        plt.show()
    
    def plot_confusion_matrices(self, filepath: str = None, figsize: Tuple = (14, 10)):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, square=True)
            
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xticklabels(['Normal', 'Fraud'])
            ax.set_yticklabels(['Normal', 'Fraud'])
        
        # Hide unused subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {filepath}")
        
        plt.show()
    
    def plot_metrics_comparison(self, filepath: str = None, figsize: Tuple = (12, 6)):
        """
        Bar plot comparing key metrics across models
        """
        df = pd.DataFrame(self.results).T[['precision', 'recall', 'f1', 'auc_roc']]
        
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {filepath}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save results to CSV"""
        df = pd.DataFrame(self.results).T
        df.to_csv(filepath)
        print(f"Results saved to {filepath}")
    
    def generate_summary_report(self) -> str:
        """Generate text summary report"""
        if not self.results:
            return "No results available"
        
        report = "\n" + "=" * 80 + "\n"
        report += "FRAUD DETECTION MODEL EVALUATION SUMMARY\n"
        report += "=" * 80 + "\n\n"
        
        # Best models per metric
        report += "BEST PERFORMERS:\n"
        report += "-" * 40 + "\n"
        
        for metric in ['precision', 'recall', 'f1', 'auc_roc']:
            best_model = max(self.results.items(), 
                            key=lambda x: x[1].get(metric, 0))[0]
            best_score = self.results[best_model][metric]
            report += f"  {metric.upper():12} → {best_model:20} ({best_score:.4f})\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "DETAILED METRICS:\n"
        report += "-" * 40 + "\n\n"
        
        for model_name, metrics in sorted(self.results.items()):
            report += f"{model_name}:\n"
            for metric, value in sorted(metrics.items()):
                report += f"  {metric:15} = {value:.4f}\n"
            report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report


class PerformanceAnalyzer:
    """Analyze model performance characteristics"""
    
    @staticmethod
    def threshold_analysis(y_true: np.ndarray, y_score: np.ndarray,
                          thresholds: np.ndarray = None) -> pd.DataFrame:
        """
        Analyze precision/recall at different thresholds
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)
        
        results = []
        
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def false_positive_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze false positives
        """
        fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
        
        return {
            'n_false_positives': len(fp_indices),
            'n_false_negatives': len(fn_indices),
            'fp_rate': len(fp_indices) / max(np.sum(y_true == 0), 1),
            'fn_rate': len(fn_indices) / max(np.sum(y_true == 1), 1),
        }
