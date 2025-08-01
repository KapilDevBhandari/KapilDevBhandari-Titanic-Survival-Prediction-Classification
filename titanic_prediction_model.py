"""
Titanic Survival Prediction
A machine learning pipeline using Logistic Regression to predict Titanic survivors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data():
    print("Reading training and test data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    passenger_ids = test['PassengerId'].copy()
    return train, test, passenger_ids

def fill_missing_values(train, test):
    print("Filling missing values...")

    age_median = train['Age'].median()
    fare_median = train['Fare'].median()
    embarked_mode = train['Embarked'].mode()[0]

    for df in [train, test]:
        df['Age'].fillna(age_median, inplace=True)
        df['Embarked'].fillna(embarked_mode, inplace=True)

    test['Fare'].fillna(fare_median, inplace=True)

def clean_data(train, test):
    print("Dropping columns not useful for prediction...")
    drop_cols = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    return train.drop(columns=drop_cols), test.drop(columns=drop_cols)

def encode_data(train, test):
    print("Encoding categorical features...")

    train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
    test = pd.get_dummies(test, columns=['Sex', 'Embarked'])

    # Align test set to match train set columns (excluding target)
    for col in train.columns:
        if col not in test.columns and col != 'Survived':
            test[col] = 0

    test = test[train.drop('Survived', axis=1).columns]
    return train, test

def get_features_targets(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test)
    )

def train_logistic_model(X, y):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate(model, X_val, y_val):
    print("\n--- Model Evaluation ---")

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds))

    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()

    return acc, roc_auc

def predict(model, X_test):
    print("Predicting on test data...")
    return model.predict(X_test)

def save_submission(ids, preds, filename='titanic_submission.csv'):
    print(f"Saving predictions to {filename}")
    pd.DataFrame({
        'PassengerId': ids,
        'Survived': preds
    }).to_csv(filename, index=False)

def main():
    print("="*40)
    print("Titanic Survival Prediction Pipeline")
    print("="*40)

    train, test, ids = load_data()
    fill_missing_values(train, test)
    train, test = clean_data(train, test)
    train, test = encode_data(train, test)
    
    X, y = get_features_targets(train)
    X_train, X_val, y_train, y_val = split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = scale(X_train, X_val, test)
    
    model = train_logistic_model(X_train_scaled, y_train)
    acc, auc_score = evaluate(model, X_val_scaled, y_val)
    
    predictions = predict(model, X_test_scaled)
    save_submission(ids, predictions)

    print("\nDone!")
    print(f"Final Accuracy: {acc:.4f}")
    print(f"Final AUC Score: {auc_score:.4f}")

if __name__ == "__main__":
    main()
