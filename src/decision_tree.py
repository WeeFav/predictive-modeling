import pandas as pd
import numpy as np
from preprocessing import EncodeX
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
import sys
import yaml
import time
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

def decision_tree(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type):
    start_time = time.time()

    with open('config/decision_tree.yaml', 'rb') as file:
        model_cfg = yaml.load(file, Loader=yaml.FullLoader)
    
    if output_column_type == 'categorical':
        assert 'target' not in model_cfg['encoding'], "cannot use target encoding when output is categorical"

    p = len(X_train.columns)
    n = len(y_test)

    transform = ColumnTransformer(transformers=[
        ('encodeX', EncodeX(), categorical_columns),
        ('scaleX', MinMaxScaler(), numerical_columns),
    ])

    #############################################################################################################
    # sklearn knn can automatically handle categorical data for output, so we don't need to encode 
    #############################################################################################################
    ############################################################################
    # the rest of the code split between whether output is numeric or categoric
    ############################################################################

    if output_column_type == 'numerical':
        pipeline = Pipeline([
            ('preprocess', transform),
            ('model', DecisionTreeRegressor())
        ])

        model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
        # print(model.get_params().keys())

        param_grid = {
            'regressor__preprocess__encodeX__encoding_method': model_cfg['encoding'],
            'regressor__model__splitter': model_cfg['splitter'],
            'regressor__model__max_depth': model_cfg['max_depth'],
            'regressor__model__min_samples_split': model_cfg['min_samples_split'],
            'regressor__model__min_samples_leaf': model_cfg['min_samples_leaf']
        }

        grid_pipeline = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=0, n_jobs=-1, error_score='raise')
        grid_pipeline.fit(X_train, y_train.values.ravel())

        best_param_dict = grid_pipeline.best_params_


        # predict
        y_pred = grid_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        score = r2 

    else:
        pipeline = Pipeline([
            ('preprocess', transform),
            ('model', DecisionTreeClassifier())
        ])        
        # print(pipeline.get_params().keys())
    
        param_grid = {
            'preprocess__encodeX__encoding_method': model_cfg['encoding'],
            'model__splitter': model_cfg['splitter'],
            'model__max_depth': model_cfg['max_depth'],
            'model__min_samples_split': model_cfg['min_samples_split'],
            'model__min_samples_leaf': model_cfg['min_samples_leaf']
        }

        grid_pipeline = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1, error_score='raise')
        grid_pipeline.fit(X_train, y_train.values.ravel())

        best_param_dict = grid_pipeline.best_params_

        # predict
        y_pred = grid_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        score = acc    
    # Save the entire pipeline
    joblib_file = "./model/decision_tree_pipeline.pkl"
    joblib.dump(grid_pipeline.best_estimator_, joblib_file)


    print("\n\nfinish decision tree")
    print("best param:", best_param_dict)
    if output_column_type == 'numerical':  
        print("r2 score:", score)
    else:
        print("accuracy:", score)
    print(f"took {time.time() - start_time} seconds")

    return score, best_param_dict, time.time() - start_time
