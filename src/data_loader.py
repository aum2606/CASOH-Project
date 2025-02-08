import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path):
        """Initialize DataLoader with path to data file.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_data(self):
        """Load data from CSV file."""
        # Read the file as text first
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line to get consistent number of columns
        processed_lines = []
        for line in lines:
            # Split by space and remove empty strings
            values = [float(v) for v in line.strip().split() if v]
            processed_lines.append(values)
        
        # Convert to numpy array
        data = np.array(processed_lines)
        
        # Assuming last column is target variable
        self.X = data[:, :-1]
        self.y = data[:, -1].reshape(-1, 1)  # Reshape for StandardScaler
        
        print(f"Loaded data shape: X={self.X.shape}, y={self.y.shape}")
        print(f"Target range: min={self.y.min():.4f}, max={self.y.max():.4f}")
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess data by scaling features and splitting into train/validation sets.
        
        Args:
            test_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Scale features and target
        self.X = self.scaler.fit_transform(self.X)
        self.y = self.target_scaler.fit_transform(self.y).ravel()  # Flatten after scaling
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Validation set shape: X={self.X_val.shape}, y={self.y_val.shape}")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def get_data(self):
        """Get processed data.
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        return self.X_train, self.X_val, self.y_train, self.y_val
