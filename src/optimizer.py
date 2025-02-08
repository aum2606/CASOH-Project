import numpy as np
from sklearn.model_selection import cross_val_score
import random

class CASOHOptimizer:
    def __init__(self, X, y, param_bounds, model_class, population_size=30, max_iterations=50):
        """Initialize CASOH optimizer.
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
            param_bounds (dict): Dictionary of parameter bounds {param_name: (min, max)}
            model_class: The class of the model to optimize
            population_size (int): Size of population for optimization
            max_iterations (int): Maximum number of iterations
        """
        self.X = X
        self.y = y
        self.param_bounds = param_bounds
        self.model_class = model_class
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.best_params = None
        self.best_score = float('-inf')
        
    def _initialize_population(self):
        """Initialize population with random values within bounds."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, bounds in self.param_bounds.items():
                if isinstance(bounds, (list, tuple)):
                    if isinstance(bounds[0], (int, float)):
                        # Numeric parameter
                        min_val, max_val = bounds
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            individual[param] = random.randint(min_val, max_val)
                        else:
                            individual[param] = random.uniform(min_val, max_val)
                    else:
                        # Categorical parameter
                        individual[param] = random.choice(bounds)
                else:
                    # Single value parameter
                    individual[param] = bounds
            population.append(individual)
        return population
    
    def _evaluate_individual(self, params):
        """Evaluate an individual's fitness using cross-validation.
        
        Args:
            params (dict): Parameters to evaluate
            
        Returns:
            float: Mean cross-validation score
        """
        model = self.model_class(**params)
        scores = cross_val_score(model, self.X, self.y, cv=5)
        return np.mean(scores)
    
    def _combined_sampling(self, population):
        """Perform combined sampling to generate new solutions.
        
        Args:
            population (list): Current population
            
        Returns:
            list: New population after combined sampling
        """
        new_population = []
        elite_size = self.population_size // 4
        
        # Sort population by fitness
        population_with_scores = [(ind, self._evaluate_individual(ind)) for ind in population]
        population_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Elite preservation
        elite = [ind for ind, _ in population_with_scores[:elite_size]]
        new_population.extend(elite)
        
        # Generate new solutions
        while len(new_population) < self.population_size:
            if random.random() < 0.5:
                # Local search
                parent = random.choice(elite)
                child = parent.copy()
                param = random.choice(list(self.param_bounds.keys()))
                bounds = self.param_bounds[param]
                
                if isinstance(bounds, (list, tuple)):
                    if isinstance(bounds[0], (int, float)):
                        # Numeric parameter
                        min_val, max_val = bounds
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            child[param] = random.randint(min_val, max_val)
                        else:
                            child[param] = random.uniform(min_val, max_val)
                    else:
                        # Categorical parameter
                        child[param] = random.choice(bounds)
                else:
                    # Single value parameter
                    child[param] = bounds
                    
                new_population.append(child)
            else:
                # Global search
                new_individual = {}
                for param, bounds in self.param_bounds.items():
                    if isinstance(bounds, (list, tuple)):
                        if isinstance(bounds[0], (int, float)):
                            # Numeric parameter
                            min_val, max_val = bounds
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                new_individual[param] = random.randint(min_val, max_val)
                            else:
                                new_individual[param] = random.uniform(min_val, max_val)
                        else:
                            # Categorical parameter
                            new_individual[param] = random.choice(bounds)
                    else:
                        # Single value parameter
                        new_individual[param] = bounds
                new_population.append(new_individual)
        
        return new_population
    
    def optimize(self):
        """Run the CASOH optimization process.
        
        Returns:
            tuple: (best_params, best_score)
        """
        population = self._initialize_population()
        
        for iteration in range(self.max_iterations):
            # Evaluate current population
            for individual in population:
                score = self._evaluate_individual(individual)
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = individual.copy()
            
            # Generate new population using combined sampling
            population = self._combined_sampling(population)
            
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
