# CASOH Implementation

This is an implementation of the Combined-Sampling Algorithm to Search the Optimized Hyperparameters (CASOH) method. The implementation is based on the research paper and provides an efficient way to optimize hyperparameters for machine learning models.

## Project Structure

```
.
├── data/
│   ├── raw.csv
│   └── test.csv
├── src/
│   ├── data_loader.py
│   ├── optimizer.py
│   ├── evaluator.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Components

1. `data_loader.py`: Handles data loading and preprocessing
2. `optimizer.py`: Implements the CASOH optimization algorithm
3. `evaluator.py`: Provides model evaluation metrics
4. `main.py`: Main script to run the program

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script from the project root directory:

```bash
python src/main.py
```

The program will:
1. Load and preprocess the data
2. Run the CASOH optimization to find the best hyperparameters
3. Train a model with the optimized parameters
4. Evaluate the model's performance

## Output

The program will display:
- Progress of the optimization process
- Best parameters found
- Model evaluation metrics (accuracy, precision, recall, F1-score)
