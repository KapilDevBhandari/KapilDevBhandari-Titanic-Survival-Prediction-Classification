# Titanic Survival Prediction Model

A complete machine learning pipeline for predicting survival on the Titanic dataset using Logistic Regression.

## Features

- **Complete Data Preprocessing**: Handles missing values, drops unnecessary columns, and encodes categorical variables
- **Feature Engineering**: One-hot encoding for categorical variables with proper column alignment
- **Model Training**: Logistic Regression with optimized hyperparameters
- **Comprehensive Evaluation**: Accuracy, classification report, confusion matrix, and ROC curve analysis
- **Visualization**: Automatic generation of confusion matrix and ROC curve plots
- **Submission Ready**: Creates Kaggle-compatible submission file

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The script expects the following files in the same directory:
- `train.csv` - Training dataset with survival labels
- `test.csv` - Test dataset for predictions

## Usage

Simply run the main script:

```bash
python titanic_prediction_model.py
```

## What the Script Does

### 1. Data Loading and Preprocessing
- Loads `train.csv` and `test.csv` files
- Handles missing values:
  - Fills missing `Age` with median
  - Fills missing `Embarked` with mode
  - Fills missing `Fare` in test data with median

### 2. Feature Engineering
- Drops unnecessary columns: `Cabin`, `Name`, `Ticket`, `PassengerId`
- Encodes categorical variables using one-hot encoding:
  - `Sex` → `Sex_female`, `Sex_male`
  - `Embarked` → `Embarked_C`, `Embarked_Q`, `Embarked_S`
- Aligns test data columns with training data

### 3. Model Training
- Splits training data into 80% train / 20% validation sets
- Scales features using StandardScaler
- Trains Logistic Regression model with:
  - `max_iter=1000`
  - `random_state=42`

### 4. Model Evaluation
- Calculates and displays accuracy score
- Generates detailed classification report
- Creates and saves confusion matrix plot
- Plots ROC curve and calculates AUC score

### 5. Predictions and Submission
- Makes predictions on test dataset
- Creates `titanic_submission.csv` file with:
  - `PassengerId` column
  - `Survived` column (predictions)

## Output Files

- `titanic_submission.csv` - Kaggle submission file
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve visualization

## Model Performance

The model typically achieves:
- **Accuracy**: ~80-85%
- **AUC Score**: ~0.80-0.85

*Note: Performance may vary due to the inherent randomness in the dataset split and model training.*

## Customization

You can easily modify the script to:
- Try different algorithms (Random Forest, SVM, etc.)
- Add feature engineering (create new features from existing ones)
- Implement cross-validation
- Use different hyperparameters
- Add more evaluation metrics

## Troubleshooting

- **Missing files**: Ensure `train.csv` and `test.csv` are in the same directory as the script
- **Import errors**: Install all required packages using `pip install -r requirements.txt`
- **Memory issues**: The script is optimized for the Titanic dataset size, but you can reduce batch sizes if needed

## License

This project is open source and available under the MIT License. 
