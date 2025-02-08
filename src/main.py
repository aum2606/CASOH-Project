from data_loader import DataLoader
from optimizer import CASOHOptimizer
from evaluator import ModelEvaluator
from models import ModelFactory
import os
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CASOH optimization for different models')
    parser.add_argument('--model', type=str, default='svm',
                      choices=ModelFactory.get_available_models(),
                      help='Model to optimize')
    parser.add_argument('--population-size', type=int, default=30,
                      help='Population size for CASOH optimization')
    parser.add_argument('--max-iterations', type=int, default=50,
                      help='Maximum number of iterations for optimization')
    parser.add_argument('--data-path', type=str, default=os.path.join('data', 'raw.csv'),
                      help='Path to the dataset')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize data loader
    print(f"Loading data from {args.data_path}...")
    data_loader = DataLoader(args.data_path)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = data_loader.load_data()
    X_train, X_val, y_train, y_val = data_loader.preprocess_data()
    
    # Get model configuration
    print(f"\nConfiguring {args.model.upper()} model...")
    model_class, param_bounds = ModelFactory.get_model_config(args.model)
    
    # Initialize and run CASOH optimizer
    print("\nStarting CASOH optimization...")
    optimizer = CASOHOptimizer(
        X_train, 
        y_train,
        param_bounds,
        model_class,
        population_size=args.population_size,
        max_iterations=args.max_iterations
    )
    
    best_params, best_score = optimizer.optimize()
    print("\nOptimization completed!")
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = ModelFactory.create_model(args.model, best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    evaluator = ModelEvaluator(final_model)
    evaluator.evaluate(X_val, y_val)
    evaluator.print_metrics()

if __name__ == "__main__":
    main()
