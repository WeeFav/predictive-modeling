import pandas as pd
import joblib
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import os
import json
from preprocessing import preprocess_data
import warnings
import keras
warnings.filterwarnings('ignore')

# model path
BASE_PATH = "C:/Users/weijlim4/Desktop/predictive-modeling/model"

# config
config_path = os.path.join(BASE_PATH, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

filepath =  config['filepath']
numerical_columns =  config['numerical_columns']
categorical_columns = config['categorical_columns']
binary_columns = config['binary_columns']
output_column_name = config['output_column_name']
output_column_type = config['output_column_type']

# get testing data
X = pd.read_csv(os.path.join(BASE_PATH, "X_test.csv"))

# If new data has already been preprocessed
# If have y data
y_true = pd.read_csv(os.path.join(BASE_PATH, "y_test.csv"))

# If new data has NOT been preprocessed
# X, *_ = preprocess_data(config, df_data=X)
# If have y data
# y_true = X.pop(output_column_name)

# create output file
# both X and y should be preprocessed
df_file = X.copy()
df_file['Actual'] = y_true

# Linear Regression
joblib_file = os.path.join(BASE_PATH, 'linear_regression_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['Linear Regression'] = predictions
    # Calculate R2 score
    r2 = r2_score(y_true, predictions)
    print("R2 score:", r2)
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())

# Logistic Regression
# Load the saved pipeline
joblib_file = os.path.join(BASE_PATH, 'logistic_regression_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['LOGISTIC_REGRESSION'] = predictions
    # Calculate accuracy
    acc = accuracy_score(y_true, predictions)
    print("Accuracy:", acc)
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())

# KNN
joblib_file = os.path.join(BASE_PATH, 'knn_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['KNN'] = predictions
    # Calculate accuracy
    accuracy = accuracy_score(y_true, predictions)
    print("Accuracy:", accuracy)
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())

# Decision Tree
joblib_file = os.path.join(BASE_PATH, 'decision_tree_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['Decision Tree'] = predictions
    if output_column_type == 'numerical':
        # Calculate R2 score
        r2 = r2_score(y_true, predictions)
        print("R2 score:", r2)
    else:
        # Calculate accuracy
        acc = accuracy_score(y_true, predictions)
        print("Accuracy:", acc)
        
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())

# Random Forest
joblib_file = os.path.join(BASE_PATH, 'random_forest_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['Random Forest'] = predictions
    if output_column_type == 'numerical':
        # Calculate R2 score
        r2 = r2_score(y_true, predictions)
        print("R2 score:", r2)
    else:
        # Calculate accuracy
        acc = accuracy_score(y_true, predictions)
        print("Accuracy:", acc)
        
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())
    
# Gradient Boosting
joblib_file = os.path.join(BASE_PATH, 'gradient_boosting_pipeline.pkl')
if os.path.exists(joblib_file):
    pipeline = joblib.load(joblib_file)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(X)
    df_file['Gradient Boosting'] = predictions
    if output_column_type == 'numerical':
        # Calculate R2 score
        r2 = r2_score(y_true, predictions)
        print("R2 score:", r2)
    else:
        # Calculate accuracy
        acc = accuracy_score(y_true, predictions)
        print("Accuracy:", acc)
        
    print('Predict: ', predictions.tolist())
    print('Actual: ', y_true.squeeze().tolist())

# Neural Network
# encode categorical data
if len(categorical_columns) > 0:
    encoderX = joblib.load(os.path.join(BASE_PATH, 'encoderX.pkl'))
    X[categorical_columns] = encoderX.transform(X[categorical_columns])
if output_column_type == 'categorical':
    encoderY = joblib.load(os.path.join(BASE_PATH, 'encoderY.pkl'))

# normalize numeric data
if len(numerical_columns) > 0:
    scalerX = joblib.load(os.path.join(BASE_PATH, 'scalerX.pkl'))
    X[numerical_columns] = scalerX.transform(X[numerical_columns])
if output_column_type == 'numerical':
    scalerY = joblib.load(os.path.join(BASE_PATH, 'scalerY.pkl'))

# load model
model = keras.models.load_model(os.path.join(BASE_PATH, 'neural_network_model.keras'))


# Prepare inputs for the model
input_df_list = [X[col] for col in categorical_columns]
numerical_df = X.drop(categorical_columns, axis=1)
input_df_list.append(numerical_df)


# Make predictions
predictions = model.predict(input_df_list, verbose=0)

# Print out prediction results and evaluate
if output_column_type == 'numerical':
    predictions = scalerY.inverse_transform(predictions) # Inverse normalization
    r2 = r2_score(y_true, predictions)
    print("R2 score:", r2)
elif output_column_type == 'categorical':
    pred_df = pd.DataFrame({f'{output_column_name}': np.argmax(predictions, axis=1)})
    predictions = encoderY.encoder.inverse_transform(pred_df) # Inverse encoding
    acc = accuracy_score(y_true, predictions)
    print("Accuracy:", acc)
else:  # binary
    y_true = y_true.to_numpy().squeeze()
    predictions = np.round(predictions.squeeze())
    acc = accuracy_score(y_true, predictions)
    print("Accuracy:", acc)

print('Predict: ', predictions.squeeze().tolist())
print('Actual: ', y_true.squeeze().tolist())
df_file['Neural Network'] = predictions

df_file.to_csv(os.path.join(BASE_PATH, 'predicted_result.csv'), index=False)
