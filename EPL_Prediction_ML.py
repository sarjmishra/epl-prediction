# Predicting EPL Football Match outcomes

# Introduction

The English Premier League (EPL) is one of the most popular and competitive football leagues globally and is watched by millions of fans from across the world. Accurately predicting the results of these matches is a challenging task, given the complex interplay of team dynamics, player performances, and external factors like home advantage. In this project, the aim is to predict EPL match outcomes based on historical match data. Analyzing historical data can reveals patterns and helps in identifying key factors influencing match outcomes.The project aims to contribute to the field of sports analytics, helping teams and analysts make data-driven decisions. By comparing the performance of Random Forest, SVM, Logistic Regression and XGBoost on EPL match predictions, this research aims to advance the accuracy and reliability of football forecasting, offering valuable insights for all stakeholders in the football ecosystem.

# Importing the required libraries
"""

# Relevant imports
from pandas import read_csv, get_dummies, DataFrame, to_datetime,Series
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""# About the Dataset

Dataset collected for this research is from https://www.football-data.co.uk/ which is an open source and provides historical football match statistics data over the years source and provides historical football match statistics data over the years. The statistics for the EPL matches spanning the last 30 seasons, starting from 1993-1994 till 2023-2024 was collected and combined into a single csv file. The dataset contains 24 columns and over 12000 records. The column ‘FTR’(Full-Time Result) indicates the outcome of a match with three values first being ‘H’(Home team wins), second being ‘A’( Away team wins) and lastly ‘D’( Draw) and is the target variable.

### Importing data from Drive
"""

# Loading the dataset
data = read_csv('/content/drive/MyDrive/Dissertation/EPL_1993_2024_Thesis_Dataset.csv')

"""### Initial Exploration of Dataset"""

# Exploring data types, missing values, data shape and columns in dataset
data.shape
print(data.info())
print(data.describe())
print(data.head())

"""# Visualising the dataset

The frequency vs. match outcome bar plot illustrates how often each type of match outcome occurs, home team wins, away team wins and draws
"""

# Target Variable Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='FTR', data=data, palette='RdYlBu')
plt.title('x')
plt.xlabel('Match Outcome')
plt.ylabel('Frequency')
plt.show()

"""The box plot shows the distribution of goals scored by home and away teams for different match outcomes: home team wins, away team wins and draws"""

# Home and Away Team Goal distribution by Match Outcome
plt.figure(figsize=(10, 6))
long_data = pd.melt(data, id_vars='FTR', value_vars=['FTHG','FTAG'],
                     var_name='Team Type', value_name='Goals')
# Create the box plot
sns.boxplot(x='FTR', y='Goals',hue='Team Type',data=long_data, palette='RdYlBu')
plt.title('Goals by Match Outcome for Home and Away Teams')
plt.xlabel('Match Outcome')
plt.ylabel('Goals')
plt.legend(title='Team Type', labels=['Home Goals (FTHG)','Away Goals (FTAG)'])
plt.show()

"""# Data Pre Processing

Extracting Year, Month, and Day from Date Column
"""

# Handling Date column and converting to year, month and day
to_datetime(392175900  , unit='s')
data['Date']=to_datetime(data['Date'],format='mixed')
data.info()
data['year']=data['Date'].dt.year
data['month']=data['Date'].dt.month
data['day']=data['Date'].dt.day

"""Removing columns that are not essential for this model or analysis"""

# dropping the irrelevant relevant columns
data = data.drop(['Div', 'Date','Referee','Time'], axis=1)

"""Removing columns that directly influence the target variable   to prevent the model from simply learning the outcomes, which can lead to overfitting and reduced generalization."""

# dropping the columns causing data leakage
data = data.drop(['FTHG','FTAG','HTR'], axis=1)

"""Removing redundant columns that convey similar information to other features in the dataset , HS(Home Shots) and AS(Away Shots) to HST (Home Shots on Target) and AST (Away shots on Target)

"""

# dropping redundant columns
data = data.drop(['HS','AS'], axis=1)
data.info()

"""Handling missing values in required columns to ensure the dataset is complete, using SimpleImputer with a mean-filling strategy

"""

# Handling missing values
columns_to_fill = ['HTAG','HTHG', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

# Creating Imputer as mean value
imputer = SimpleImputer(strategy='mean')

# Applying imputer to the selected columns
data[columns_to_fill] = imputer.fit_transform(data[columns_to_fill])
data.info()

"""Encoding FTR - Target Variable"""

# Checking unique values in Target column
data['FTR'].unique()

# Binary Mapping of values in Target column
data['FTR'] = data['FTR'].map({'H': 1, 'D': 0, 'A': 0})

"""Using one-hot encoding to the convert categorical values for columns HomeTeam and AwayTeam into numerical values with drop first as true for dimensionality reduction.

"""

# One-hot encoding for Home Team and Away Team
data = pd.get_dummies(data, columns=['HomeTeam', 'AwayTeam'], drop_first=True)
print(data.info())
print(data.head(2))

"""# Predictive Modeling

## Dependent and Independent Variables

**Dependent Variable** - FTR (Full Time Result)

**Independent Variable** - All of other features taken from  the data set

Using a Binary class mapping for Home Win, Draw and Away Win for Target Variable Y considering Home Win as True Positive and Draw and Away Win as True negative
"""

# Dividing dataset into label and feature
X = data.drop(['FTR'], axis=1)
y = data['FTR']
print(X.shape)
print(y.shape)

# Heatmap to establish correlation and causation
correlation  = X.corr()
N = 20
top_features = correlation.abs().mean().sort_values(ascending=False).head(N).index  # Top N most correlated on average
filtered_corr = correlation.loc[top_features, top_features]

figure = ff.create_annotated_heatmap(filtered_corr.values,list(filtered_corr.columns),list(filtered_corr.columns),filtered_corr.round(2).values,showscale=True)
figure.show()

"""The heatmap shows that none of the features show strong correlation or causation, suggesting that no features can be dropped based on this.

Scaling the dataset for normalisation
"""

# Data scaling
X_scaled = StandardScaler().fit_transform(X)
DataFrame(X_scaled)

"""Spliting the dataset into train and test sets with a ratio of 80:20

"""

# Data splitting into train and test set : Classical 80-20 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.20, random_state=100)
# Data shape after train test split
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""Balancing the training set using SMOTE"""

# Data Balancing using SMOTE
X_train, y_train = SMOTE(random_state=100).fit_resample(X_train, y_train)
#Training data shape after balancing
print(X_train.shape, y_train.value_counts())

"""# Binary Classification

Implementing various supervised learning models to predict outcomes based on labeled binary target variables. The models used are Random Forest, Logistic Regression, Support Vector Machine (SVM), and XGBoost

## Model Building and Validation

### Random Forest Classifer

Buiding Random Forest Classifier
"""

# Random Forest Pipeline
rf_classifier = Pipeline([('classification', RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=1))])

# Hyperparameter tuning grid for RandomForestClassifier
no_trees= {
    'classification__n_estimators': [50, 100, 150, 200, 300, 400, 500, 600, 700],  # Number of trees
}
# no_trees = {'classification__n_estimators': [10, 30, 50, 70, 100, 150, 200, 300, 400, 500]}
# no_trees = {'classification__n_estimators': [50, 100, 150, 200, 250, 300]}
# no_trees = {'classification__n_estimators': [10, 20, 30, 40, 50, 60]}


# Perform GridSearchCV with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=no_trees, scoring='accuracy', cv=5)
grid_search_rf.fit(X_train, y_train)  # Training, testing, evaluating, ranking

# Best parameters for Random Forest
best_params_rf = grid_search_rf.best_params_
print("Best parameters for Random Forest:", best_params_rf)

# Best accuracy score from the cross-validation process
best_accuracy_rf = grid_search_rf.best_score_
print("Best cross-validation Accuracy:", best_accuracy_rf)

best_rf_model = grid_search_rf.best_estimator_  # Saving the best model from the pipeline

"""Incorporating the results of hyper-parameters from hyperparameter tuning in the above step and building the Random forest classifier with best parameter
{'classification__n_estimators': 600 }

Prediction on Test Set with model created with best hyperparameter and finding the important features
"""

rf_classifier1 = RandomForestClassifier(n_estimators= 600, criterion='entropy', max_features='sqrt', random_state=1)  # building model
rf_classifier1.fit(X_train,y_train) #training
y_pred_rf = rf_classifier1.predict(X_test) # testing
print("Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Test Recall :", recall_score(y_test, y_pred_rf))
print("Test Precision :", precision_score(y_test, y_pred_rf))
print("Test F1 Score :", f1_score(y_test, y_pred_rf))
# printing important features
imp_features = Series(rf_classifier1.feature_importances_, index=list(X.columns)).sort_values(ascending=False)
print(imp_features.head(15))

"""**Observation** - The accuracy of the training and the test set is almost similar around 76%, indicating that the model is not over fitting. The best features that Random Forest gave are HTHG, HTAG, day, HST year, month and AST

Building Random forest classifer with the best features.
"""

# Feature selection
X1 = data[['HTHG', 'HTAG', 'day', 'HST', 'year', 'month', 'AST']]  # Selected features
# Scale the features
X_scaled_imp = StandardScaler().fit_transform(X1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_imp, y, test_size=0.2, random_state=42, stratify=y
)
# Random Forest Classifier Pipeline
rf_classifier_imp_features = Pipeline([
    ('balancing', SMOTE(random_state=101)),
    ('classification', RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=1))
])

# Hyperparameter tuning grid for RandomForestClassifier
no_trees = {
    'classification__n_estimators': [10, 50, 100, 150, 200, 300, 400, 500],  # Number of trees
}

# Perform GridSearchCV with cross-validation
grid_search_rf_best = GridSearchCV(estimator=rf_classifier_imp_features, param_grid=no_trees, scoring='accuracy', cv=5)
grid_search_rf_best.fit(X_train, y_train)  # Train the model using the training set

# Best parameters for Random Forest
best_params_rf_best = grid_search_rf_best.best_params_
print("Best parameters for Random Forest:", best_params_rf_best)

# Best accuracy score from the cross-validation process
best_accuracy_rf_best = grid_search_rf_best.best_score_
print("Best cross-validation Accuracy:", best_accuracy_rf_best)

best_rf_model_imp = grid_search_rf_best.best_estimator_  # Saving the best model from the pipeline

"""Prediction on Test Set with model created with best hyperparameter"""

rf_classifier_imp_features1 = RandomForestClassifier(n_estimators= 400, criterion='entropy', max_features='sqrt', random_state=1)  # building model
rf_classifier_imp_features1.fit(X_train,y_train) #training
y_pred_rf_imp = rf_classifier_imp_features1.predict(X_test) # testing
print("Test Accuracy:", accuracy_score(y_test, y_pred_rf_imp))
print("Test Recall :", recall_score(y_test, y_pred_rf_imp))
print("Test Precision :", precision_score(y_test, y_pred_rf_imp))
print("Test F1 Score :", f1_score(y_test, y_pred_rf_imp))

"""**Observation** - The accuracy of the training and the test set is almost similar around ~74% respectively indicating that the model is not over fitting

### Logistic Regression

Buiding Logistic Regression Classifier
"""

# Pipeline for SGDClassifier
sgd_classifier = Pipeline([('classification', SGDClassifier(loss ='log_loss', penalty ='elasticnet', random_state = 1))])

# Hyperparameter tuning
grid_param_lr = {
    'classification__eta0': [.001, .01, .1, 1, 10, 100],
    'classification__max_iter': [50,100,150,200,250],
    'classification__alpha': [.001, .01, .1, 1, 10, 100],
    'classification__l1_ratio': [0, 0.3, 0.5, 0.7, 1]
}

# GridSearchCV with cross-validation
grid_search_lr = GridSearchCV(estimator=sgd_classifier, param_grid=grid_param_lr, scoring='accuracy', cv=5)

# Fit the grid search
grid_search_lr.fit(X_train, y_train) # training, testing, evaluating, ranking

# Results for binary classification
best_params_lr = grid_search_lr.best_params_
print("Best parameters Logistic Regression:", best_params_lr)

# Best scores for each metric at the best hyperparameters
best_accuracy_lr = grid_search_lr.best_score_  # The best score according to the refit metric accuracy
print("Best cross-validation Accuracy:", best_accuracy_lr)

best_sgd_model = grid_search_lr.best_estimator_  # Saving the best model from the pipeline

"""Prediction on Test Set with model created with best hyperparameter"""

sgd_classifier1 = SGDClassifier(alpha=0.01, eta0= 0.001, l1_ratio = 0.5, max_iter = 50, random_state=1)  # building model
sgd_classifier1.fit(X_train,y_train) #training
y_pred_sgd= sgd_classifier1.predict(X_test) # testing
print("Test Accuracy:", accuracy_score(y_test, y_pred_sgd))
print("Test Recall :", recall_score(y_test, y_pred_sgd))
print("Test Precision :", precision_score(y_test, y_pred_sgd))
print("Test F1 Score :", f1_score(y_test, y_pred_sgd))

"""**Observation** - The accuracy of the training and the test set is almost similar around 76.20% and 75.03 % respectively indicating that the model is not over fitting

### Support Vector Classifier

Buiding Support Vector Machine Classifier
"""

# Pipeline for SVC
svc_classifier = Pipeline([('classification', SVC(random_state=1))])

# Hyperparameter tuning grid for SVC with linear kernel
grid_param_svc = {
    'classification__kernel': ['linear', 'rbf', 'poly','sigmoid'],
    'classification__C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
}

# GridSearchCV with cross-validation
grid_search_svc = GridSearchCV(estimator=svc_classifier, param_grid=grid_param_svc, scoring='accuracy', cv=5)

# Fit the grid search
grid_search_svc.fit(X_train, y_train)  # Training, testing, evaluating, ranking

# Results for binary classification
best_params_svc = grid_search_svc.best_params_
print("Best parameters SVC:", best_params_svc)

# Best scores for each metric at the best hyperparameters
best_accuracy_svc = grid_search_svc.best_score_  # The best score according to the refit metric accuracy
print("Best cross-validation Accuracy:", best_accuracy_svc)

best_svc_model = grid_search_svc.best_estimator_  # Saving the best model from the pipeline

"""Prediction on Test Set with model created with best hyperparameter"""

svc_classifier1 = SVC(kernel= 'linear', C = 10, random_state=1)  # building model
svc_classifier1.fit(X_train,y_train) #training
y_pred_svc= svc_classifier1.predict(X_test) # testing
print("Test Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Test Recall :", recall_score(y_test, y_pred_svc))
print("Test Precision :", precision_score(y_test, y_pred_svc))
print("Test F1 Score :", f1_score(y_test, y_pred_svc))

"""**Observation** - The accuracy of the training and the test set is almost similar around 75.87% and 75.40% respectively indicating that the model is not overfitting.

### XGBoost

Building XGBoost classifier
"""

# Define the pipeline with SMOTE and the XGBoost classifier
xgb_classifier = Pipeline([('classification', XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'))])

# Hyperparameter tuning grid
xgb_params = {
    'classification__n_estimators': [50, 100, 150, 200],
    'classification__max_depth': [3, 5, 7],
    'classification__learning_rate': [0.01, 0.05, 0.1],
    'classification__subsample': [0.8, 0.9, 1.0],  # Subsample ratio of the training instances
    'classification__colsample_bytree': [0.7, 0.8, 1.0]
}

# GridSearchCV with cross-validation
grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_params,scoring='accuracy', cv= 5)

# Fit the grid search to the training data
grid_search_xgb.fit(X_train, y_train)

# Get the best hyperparameters
best_params_xgb = grid_search_xgb.best_params_
print("Best parameters XGBoost:", best_params_xgb)

# Best cross-validation scores for each metric
best_accuracy_xgb = grid_search_xgb.best_score_
print("Best cross-validation Accuracy:", best_accuracy_xgb)

best_xgb_model = grid_search_xgb.best_estimator_  # Saving the best model from the pipeline

"""Prediction on Test Set with model created with best hyperparameter"""

xgb_classifier1 = XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.1, max_depth= 5,n_estimators = 50,subsample =0.8, random_state=1)  # building model
xgb_classifier1.fit(X_train,y_train) #training`
y_pred_xgb = xgb_classifier1.predict(X_test) # testing
print("Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Test Recall :", recall_score(y_test, y_pred_xgb))
print("Test Precision :", precision_score(y_test, y_pred_xgb))
print("Test F1 Score :", f1_score(y_test, y_pred_xgb))

"""**Observation** - The accuracy of the training and the test set is almost similar around 75.67% and 74.62% respectively indicating that the model is not overfitting.

# Result

After evaluating the performance of all the selected four machine learning models—Random Forest, Logistic Regression, Support Vector Machine (SVM) and XGBoost, it was observed that Random Forest Classifier achieved the highest test accuracy at 75.70%, demonstrating its robust performance. Support Vector Machine (SVM) followed closely with an accuracy of 75.40%, while Logistic Regression achieved a test accuracy of 75.03%. XGBoost, despite its consistent performance, achieved a slightly lower accuracy of 74.62%. These results highlight the strong predictive capabilities of all models, with Random Forest being the most accurate for this particular task of predicting football match outcome. However, all models demonstrated a reasonable level of accuracy, indicating their potential for effective football match outcome predictions.

# Making Predcitions on Unseen data from Season 2024

Applying the trained machine learning model to predict outcomes for unseen data from the 2024 season
"""

# Loading the dataset
new_data = read_csv('/content/drive/MyDrive/Dissertation/Unseen_Data_2024.csv')
new_data.info()

"""## Pre Processing the unseen data"""

# Loading the unseen data
unseen_data = pd.read_csv('/content/drive/MyDrive/Dissertation/Unseen_Data_2024.csv')

# Preprocessing unseen data similar to training data
unseen_data['Date'] = pd.to_datetime(unseen_data['Date'], format='mixed')
unseen_data['year'] = unseen_data['Date'].dt.year
unseen_data['month'] = unseen_data['Date'].dt.month
unseen_data['day'] = unseen_data['Date'].dt.day
unseen_data = unseen_data.drop(['Div', 'Date', 'Referee', 'Time'], axis=1)

# Imputing missing values
columns_to_fill = ['HTAG', 'HTHG', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
unseen_data[columns_to_fill] = imputer.transform(unseen_data[columns_to_fill])

# One-hot encode HomeTeam and AwayTeam
unseen_data = pd.get_dummies(unseen_data, columns=['HomeTeam', 'AwayTeam'], drop_first=True)

"""## Making Predictions"""

# Aligning columns with training data
for col in X.columns:  # X.columns is from training data
    if col not in unseen_data.columns:
        unseen_data[col] = 0  # Add missing columns
unseen_data = unseen_data[X.columns]  # Ensure same column order

# Scaling the unseen data
unseen_data_scaled = StandardScaler().fit_transform(unseen_data)

# Making predictions
predictions = best_rf_model.predict(unseen_data_scaled)

# Mapping predictions back to categorical values
mapping = {1: 'H', 0: 'A'}  # '1' = Home Win, '0' = Draw Away Win
predicted_categories = [mapping.get(pred, 'A') for pred in predictions]

# Adding predictions to the unseen_data DataFrame
unseen_data['Predicted_Result'] = predicted_categories

# Extracting team names (HomeTeam and AwayTeam) from the encoded columns
home_team_columns = [col for col in unseen_data.columns if col.startswith('HomeTeam_')]
away_team_columns = [col for col in unseen_data.columns if col.startswith('AwayTeam_')]

# Extracting the actual team names by selecting the column with '1'
unseen_data['HomeTeam'] = unseen_data[home_team_columns].idxmax(axis=1).str.replace('HomeTeam_', '')
unseen_data['AwayTeam'] = unseen_data[away_team_columns].idxmax(axis=1).str.replace('AwayTeam_', '')

# Preparing the final DataFrame with HomeTeam, AwayTeam, and Prediction columns
output_data = unseen_data[['HomeTeam', 'AwayTeam', 'Predicted_Result']]

# Displaying the result
print("Prediction Results:")
display(output_data)

# Saving the results to a new CSV file
output_data.to_csv('predicted_results.csv', index=False)

"""# Conclusion

This study aimed to improve the accuracy of football match outcome prediction in the English Premier League (EPL) using machine learning models like Random Forest, Logistic Regression, SVM, and XGBoost, analyzing data from 1993 to 2024. After employing techniques such as feature selection, SMOTE for data balancing, and dimensionality reduction, the models achieved competitive performance in regrads to previous researches, with Random Forest yielding the highest accuracy of 75.70%.
"""
