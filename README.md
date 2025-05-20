# Computing Individual Research Project

This repository contains the code and analysis for a data science research project focusing on machine learning and data visualization.

## Data source
https://physionet.org/content/mimiciv/3.1/
Due to huge data size the actual data is not uploaded in the github.

## Project Structure

```
├── data/               # Data directory
├── src/               # Source code
│   ├── features/      # Feature engineering code
│   ├── models/        # Machine learning models
│   ├── utils/         # Utility functions
│   ├── visualization/ # Data visualization code
│   └── main.ipynb     # Main analysis notebook
└── requirements.txt   # Project dependencies
```

## Requirements

The project requires Python 3.x and the following main dependencies:
- pandas (>=1.3.0)
- numpy (>=1.20.0)
- matplotlib (>=3.4.0)
- seaborn (>=0.11.0)
- scikit-learn (>=0.24.0)
- jupyter (>=1.0.0)
- notebook (>=6.4.0)
- ipykernel (>=6.0.0)
- xgboost>=1.5.0
- lightgbm>=3.3.0
- imbalanced-learn>=0.8.0
- comorbidipy>=0.1.0

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rupace10/STW7048CEM_Computing_Individual_Research_Project.git
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the project directory
2. Launch Jupyter Notebook:
```bash
jupyter notebook
```
3. Open `src/main.ipynb` to run the analysis

## Project Components

- **Data Processing**: Data preprocessing and feature engineering code in the `features` directory
- **Models**: Machine learning model implementations in the `models` directory
- **Visualization**: Data visualization code in the `visualization` directory
- **Utilities**: Helper functions and utilities in the `utils` directory

## Results

Model performance metrics are stored in `src/model_performance_summary.csv`.


## Author

Rupesh Maharjan