import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path):
        """Initialize model trainer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train multiple models and select the best one based on performance."""
        try:
            mlflow.set_experiment("bank_churn_prediction")
            
            best_score = 0
            best_model = None
            
            for algorithm in self.config['model']['algorithms']:
                with mlflow.start_run(nested=True):
                    model = self._get_model(algorithm)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Log parameters and metrics
                    mlflow.log_params(algorithm['params'])
                    mlflow.log_metrics(metrics)
                    
                    # Update best model if necessary
                    if metrics['f1'] > best_score:
                        best_score = metrics['f1']
                        best_model = model
                        
                    logger.info(f"Trained {algorithm['name']} - F1 Score: {metrics['f1']:.4f}")
            
            # Save best model
            self._save_model(best_model)
            return best_model, best_score
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _get_model(self, algorithm):
        """Get model instance based on algorithm configuration."""
        if algorithm['name'] == 'logistic_regression':
            return LogisticRegression(**algorithm['params'])
        elif algorithm['name'] == 'random_forest':
            return RandomForestClassifier(**algorithm['params'])
        elif algorithm['name'] == 'xgboost':
            return xgb.XGBClassifier(**algorithm['params'])
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm['name']}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate model performance metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def _save_model(self, model):
        """Save the trained model to disk."""
        import os
        model_path = self.config['model']['best_model_path']
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")