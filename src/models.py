from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ModelFactory:
    """Factory class for creating and configuring different models with their parameter bounds."""
    
    @staticmethod
    def get_model_config(model_name):
        """Get model and its parameter bounds for optimization.
        
        Args:
            model_name (str): Name of the model to configure
            
        Returns:
            tuple: (model_class, parameter_bounds)
        """
        configs = {
            'svm': {
                'model': SVR,
                'params': {
                    'C': (0.1, 100.0),
                    'gamma': (0.001, 10.0),
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },
            'rf': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'random_state': 42  # Fixed parameter
                }
            },
            'gbm': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 10),
                    'min_samples_split': (2, 20),
                    'random_state': 42
                }
            },
            'mlp': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'learning_rate_init': (0.0001, 0.1),
                    'max_iter': (100, 500),
                    'alpha': (0.0001, 0.01),
                    'random_state': 42
                }
            },
            'knn': {
                'model': KNeighborsRegressor,
                'params': {
                    'n_neighbors': (1, 20),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # Manhattan or Euclidean distance
                }
            },
            'xgb': {
                'model': XGBRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.5, 1.0),
                    'colsample_bytree': (0.5, 1.0),
                    'random_state': 42
                }
            },
            'lgbm': {
                'model': LGBMRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 100),
                    'subsample': (0.5, 1.0),
                    'random_state': 42
                }
            }
        }
        
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(configs.keys())}")
            
        return configs[model_name]['model'], configs[model_name]['params']
    
    @staticmethod
    def create_model(model_name, params=None):
        """Create a model instance with given parameters.
        
        Args:
            model_name (str): Name of the model to create
            params (dict, optional): Parameters for the model
            
        Returns:
            object: Model instance
        """
        model_class, _ = ModelFactory.get_model_config(model_name)
        if params is None:
            return model_class()
        return model_class(**params)
    
    @staticmethod
    def get_available_models():
        """Get list of available model names.
        
        Returns:
            list: List of available model names
        """
        return ['svm', 'rf', 'gbm', 'mlp', 'knn', 'xgb', 'lgbm']
