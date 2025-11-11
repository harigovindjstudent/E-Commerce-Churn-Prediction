import logging
import yaml
import mlflow
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineering
from src.models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    try:
        # Load configuration
        config_path = '../config/config.yaml'
        
        # Initialize components
        data_loader = DataLoader(config_path)
        feature_engineer = FeatureEngineering(config_path)
        model_trainer = ModelTrainer(config_path)
        
        # Load and split data
        logger.info("Loading data...")
        df = data_loader.load_data()
        X_train, X_test, y_train, y_test = data_loader.split_data(df)
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        X_train = feature_engineer.create_features(X_train)
        X_test = feature_engineer.create_features(X_test)
        
        # Preprocess features
        X_train = feature_engineer.preprocess_features(X_train, is_training=True)
        X_test = feature_engineer.preprocess_features(X_test, is_training=False)
        
        # Train model
        logger.info("Training models...")
        best_model, best_score = model_trainer.train_model(X_train, X_test, y_train, y_test)
        
        logger.info(f"Training completed successfully. Best F1 Score: {best_score:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()