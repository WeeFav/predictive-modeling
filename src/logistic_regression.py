import pandas as pd
import numpy as np
from preprocessing import EncodeX
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

def logistic_regression(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type):
    start_time = time.time()

    # logistic regression only for binary output
    assert output_column_type == 'binary', 'output must be binary to use logistic regression'

    with open('config/logistic_regression.yaml', 'rb') as file:
        model_cfg = yaml.load(file, Loader=yaml.FullLoader)

    p = len(X_train.columns)
    n = len(y_test)

    transform = ColumnTransformer(transformers=[
        ('encodeX', EncodeX(), categorical_columns),
        ('scaleX', MinMaxScaler(), numerical_columns),
    ])
    
    pipeline = Pipeline([
        ('preprocess', transform),
        ('model', LogisticRegression())
    ])    

    # print(pipeline.get_params().keys())

    param_grid = {
        'preprocess__encodeX__encoding_method': model_cfg['encoding'],
        'model__solver': model_cfg['solver'],
        'model__C': model_cfg['C']
    }

    grid_pipeline = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1, error_score='raise')
    grid_pipeline.fit(X_train, y_train.values.ravel())

    best_param_dict = grid_pipeline.best_params_

    # Save the entire pipeline
    joblib_file = "./model/logistic_regression_pipeline.pkl"
    joblib.dump(grid_pipeline.best_estimator_, joblib_file)

    # Predict
    y_pred = grid_pipeline.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    print("\n\nfinish logistic regression")
    print("Best parameters:", best_param_dict)
    print("Accuracy:", acc)
    print(f"Took {time.time() - start_time} seconds")

    return acc, best_param_dict, time.time() - start_time
