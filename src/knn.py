import pandas as pd
import numpy as np
from preprocessing import EncodeX
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import yaml
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

def knn(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type):
    start_time = time.time()

    # knn only for categoric output
    assert output_column_type == 'categorical', 'output must be categorical to use knn'

    with open('config/knn.yaml', 'rb') as file:
        model_cfg = yaml.load(file, Loader=yaml.FullLoader)

    assert 'target' not in model_cfg['encoding'], "cannot use target encoding when output is categorical"

    p = len(X_train.columns)
    n = len(y_test)

    transform = ColumnTransformer(transformers=[
        ('encodeX', EncodeX(), categorical_columns),
        ('scaleX', MinMaxScaler(), numerical_columns),
    ])
    
    pipeline = Pipeline([
        ('preprocess', transform),
        ('model', KNeighborsClassifier())
    ])

    ##################################################################################################
    # sklearn knn can automatically handle categorical data for output, so we don't need to encode 
    ##################################################################################################


    # print(pipeline.get_params().keys())

    n_neighbors_list = range(1, model_cfg['n_neighbors_range'] + 1)

    param_grid = {
        'preprocess__encodeX__encoding_method': model_cfg['encoding'],
        'model__weights': model_cfg['weights'],
        'model__metric': model_cfg['metric'],
        'model__n_neighbors': n_neighbors_list
    }

    grid_pipeline = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1, error_score='raise')
    grid_pipeline.fit(X_train, y_train.values.ravel())

    best_param_dict = grid_pipeline.best_params_

    # Save the entire pipeline
    joblib_file = "./model/knn_pipeline.pkl"
    joblib.dump(grid_pipeline.best_estimator_, joblib_file)

    # Predict
    y_pred = grid_pipeline.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    print("\n\nfinish KNN")
    print("Best parameters:", best_param_dict)
    print("Accuracy:", acc)
    print(f"Took {time.time() - start_time} seconds")

    return acc, best_param_dict, time.time() - start_time
