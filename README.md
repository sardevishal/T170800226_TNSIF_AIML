# Heart Disease Prediction using Machine Learning

## Overview

This project uses machine learning to predict whether a patient is likely to have heart disease based on clinical and health-related features.

The project was developed in Google Colab and includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Train/test split
- Multiple machine learning models
- Model evaluation
- ROC-AUC comparison
- 5-fold cross-validation
- Random Forest hyperparameter tuning
- An interactive prediction interface using `ipywidgets`

## Dataset

The project uses a CSV dataset named:

```text
heart_disease_dataset.csv
```

The target column is:

```text
heart_disease
```

where:

- `0` = No heart disease
- `1` = Heart disease

The input features used by the prediction interface are:

- `age`
- `sex`
- `chest_pain_type`
- `resting_blood_pressure`
- `cholesterol`
- `fasting_blood_sugar`
- `resting_ecg`
- `max_heart_rate`
- `exercise_induced_angina`
- `st_depression`
- `st_slope`
- `num_major_vessels`
- `thalassemia`

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- IPyWidgets

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv('/content/heart_disease_dataset.csv')
```

### 2. Exploratory Data Analysis

The project examines:

- Dataset shape
- Data types
- Statistical summary
- Missing values
- Target distribution
- Feature distributions
- Skewness
- Correlation matrix
- Relationship between important features and heart disease

Visualizations include count plots, box plots, and a correlation heatmap.

### 3. Data Splitting

The dataset is divided into training and testing sets using an 80/20 split.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### 4. Machine Learning Models

The following models are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

### 5. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- ROC-AUC
- Confusion Matrix
- Classification Report

A model performance comparison and ROC curve comparison are also generated.

## Model Performance

The notebook contains the following recorded test-set results:

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6750 | 0.6957 | 0.7273 | 0.7111 | 0.6111 | 0.7342 |
| Decision Tree | 0.5750 | 0.6000 | 0.6818 | 0.6383 | 0.4444 | 0.5631 |
| Random Forest | 0.6625 | 0.6735 | 0.7500 | 0.7097 | 0.5556 | 0.7601 |
| SVM | 0.6500 | 0.6739 | 0.7045 | 0.6889 | 0.5833 | 0.7355 |

Based on the recorded results, Random Forest has the highest ROC-AUC among the four baseline models.

## Cross-Validation

A 5-fold Stratified Cross-Validation is performed on the Random Forest model.

The evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Random Forest Hyperparameter Tuning

Randomized Search is used to tune the Random Forest model.

The searched parameters include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

The search evaluates 20 parameter combinations using 5-fold cross-validation and optimizes for F1-score.

The best estimator is stored as:

```python
best_rf
```

This is the final tuned Random Forest model used by the interactive prediction interface.

## Interactive Prediction

The notebook provides an interactive UI using `ipywidgets`.

Users can enter patient information such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST slope
- Number of major vessels
- Thalassemia

After clicking the **Predict** button, the system displays:

- Predicted result
- Probability of heart disease

The prediction uses:

```python
prediction = best_rf.predict(input_data)[0]
```

and:

```python
probability = best_rf.predict_proba(input_data)[0][1]
```

## Saving the Trained Model

The trained model used in the prediction interface is named `best_rf`, not `model`.

To save it as a `.pkl` file:

```python
import pickle

with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(best_rf, file)

print("Model saved successfully!")
```

To download it from Google Colab:

```python
from google.colab import files

files.download("heart_disease_model.pkl")
```

The resulting file can be used later to load the trained model without retraining it.

To load the model:

```python
import pickle

with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)
```

## How to Run the Project

### Option 1: Google Colab

1. Open the project notebook in Google Colab.
2. Upload `heart_disease_dataset.csv` to the Colab runtime.
3. Run the notebook cells from top to bottom.
4. Wait for model training and hyperparameter tuning to finish.
5. Use the interactive prediction interface.

### Option 2: Local Python Environment

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

Place the dataset in the appropriate project directory and update the CSV path if necessary.

Then run the Python notebook/script.

## Project Structure

```text
Heart-Disease-Prediction/
│
├── group_1.py
├── heart_disease_dataset.csv
├── heart_disease_model.pkl
└── README.md
```

`heart_disease_model.pkl` is generated after running the model-saving code.

## Important Note

This project is a machine learning prediction project for educational and demonstration purposes. Its predictions should not be treated as a medical diagnosis or as a replacement for professional medical advice.

## Author

Heart Disease Prediction ML Project
