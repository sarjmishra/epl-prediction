
# Predicting EPL Football Match Outcomes

## Introduction

The English Premier League (EPL) is one of the most popular and competitive football leagues globally and is watched by millions of fans from across the world. Accurately predicting the results of these matches is a challenging task, given the complex interplay of team dynamics, player performances, and external factors like home advantage. In this project, the aim is to predict EPL match outcomes based on historical match data. Analyzing historical data can reveals patterns and helps in identifying key factors influencing match outcomes. The project aims to contribute to the field of sports analytics, helping teams and analysts make data-driven decisions. By comparing the performance of Random Forest, SVM, Logistic Regression and XGBoost on EPL match predictions, this research aims to advance the accuracy and reliability of football forecasting, offering valuable insights for all stakeholders in the football ecosystem.

## Importing the Required Libraries

```python
# Relevant imports
from pandas import read_csv, get_dummies, DataFrame, to_datetime, Series
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

## About the Dataset

Dataset collected for this research is from [Football Data](https://www.football-data.co.uk/) which is an open source and provides historical football match statistics data over the years. The statistics for the EPL matches spanning the last 30 seasons, starting from 1993-1994 till 2023-2024 was collected and combined into a single csv file. The dataset contains 24 columns and over 12000 records. The column ‘FTR’(Full-Time Result) indicates the outcome of a match with three values first being ‘H’(Home team wins), second being ‘A’( Away team wins) and lastly ‘D’( Draw) and is the target variable.

## Preprocessing the Data

### Handling Date Column

Converting the Date column to extract year, month, and day information:

```python
data['Date'] = pd.to_datetime(data['Date'], format='mixed')
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
```

### Dropping Irrelevant Columns

Dropping irrelevant columns such as 'Div', 'Date', 'Referee', 'Time', and handling missing values:

```python
data = data.drop(['Div', 'Date','Referee','Time'], axis=1)
```

### Encoding the Target Variable

Mapping the target variable 'FTR' to binary values:

```python
data['FTR'] = data['FTR'].map({'H': 1, 'D': 0, 'A': 0})
```

### One-hot Encoding

One-hot encoding HomeTeam and AwayTeam:

```python
data = pd.get_dummies(data, columns=['HomeTeam', 'AwayTeam'], drop_first=True)
```

## Predictive Modeling

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=100)
```

### Data Balancing using SMOTE

```python
X_train, y_train = SMOTE(random_state=100).fit_resample(X_train, y_train)
```

## Models Used

- **Random Forest**: 
```python
rf_classifier = Pipeline([('classification', RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=1))])
```

- **Logistic Regression**: 
```python
sgd_classifier = Pipeline([('classification', SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=1))])
```

- **Support Vector Classifier**: 
```python
svc_classifier = Pipeline([('classification', SVC(random_state=1))])
```

- **XGBoost**: 
```python
xgb_classifier = Pipeline([('classification', XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'))])
```

## Evaluation Metrics

The model performance is evaluated based on the following metrics:

- Accuracy
- Recall
- Precision
- F1-Score

## Results

After evaluating the performance of all the selected four machine learning models—Random Forest, Logistic Regression, Support Vector Machine (SVM), and XGBoost, it was observed that:

- **Random Forest**: 75.70%
- **Support Vector Machine (SVM)**: 75.40%
- **Logistic Regression**: 75.03%
- **XGBoost**: 74.62%

Random Forest classifier achieved the highest test accuracy, demonstrating its robust performance.

## Making Predictions on Unseen Data

The trained models are applied to predict outcomes for unseen data from the 2024 season.

```python
new_data = read_csv('/content/drive/MyDrive/Dissertation/Unseen_Data_2024.csv')
unseen_data = unseen_data.drop(['Div', 'Date', 'Referee', 'Time'], axis=1)
```

## Conclusion

This project demonstrates the strong predictive capabilities of machine learning models for predicting football match outcomes. The Random Forest model emerged as the most accurate, but other models such as SVM, Logistic Regression, and XGBoost also provided valuable insights.
