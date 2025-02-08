from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ModelEvaluator:
    def __init__(self, model):
        """Initialize ModelEvaluator with a trained model.
        
        Args:
            model: Trained model instance
        """
        self.model = model
        self.metrics = {}
        
    def evaluate(self, X_val, y_val):
        """Evaluate model performance on validation data.
        
        Args:
            X_val (array-like): Validation features
            y_val (array-like): Validation target values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X_val)
        
        self.metrics['mse'] = mean_squared_error(y_val, y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(y_val, y_pred)
        self.metrics['r2'] = r2_score(y_val, y_pred)
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics."""
        if not self.metrics:
            print("No metrics available. Run evaluate() first.")
            return
        
        print("\nModel Evaluation Metrics:")
        print("-" * 30)
        print(f"MSE: {self.metrics['mse']:.4f}")
        print(f"RMSE: {self.metrics['rmse']:.4f}")
        print(f"MAE: {self.metrics['mae']:.4f}")
        print(f"R²: {self.metrics['r2']:.4f}")
        print("-" * 30)
