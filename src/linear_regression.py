import pandas as pd
import numpy as np
from preprocessing import EncodeX
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import yaml
import time
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

def linear_regression(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type):
    start_time = time.time()

    # linear regression only for numerical output
    assert output_column_type == 'numerical', 'output must be numeric to use linear regression'

    # get hyperparameters
    with open('config/linear_regression.yaml', 'rb') as file:
        model_cfg = yaml.load(file, Loader=yaml.FullLoader)

    # variables used for calculating adjusted R2
    p = len(X_train.columns)
    n = len(y_test)

    # ColumnTransformer used for transforming input columns
    transform = ColumnTransformer(transformers=[
        ('encodeX', EncodeX(), categorical_columns),       # encode categoric columns, use customized class
        ('scaleX', MinMaxScaler(), numerical_columns),     # normalize/standardlize numeric columns
    ])
    
    pipeline = Pipeline([
        ('preprocess', transform),
        ('polyFeatures', PolynomialFeatures()),     # transform features to polynomial features
                                                    # if degree = 1, then transformed features will be the same as original features
                                                    # else, new features (x1^2, x1*x2^2, etc.) will be added
        ('model', LinearRegression())
    ])    

    # TransformedTargetRegressor is used for transforming output column
    model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())

    # print(model.get_params().keys())

    # specify hyperparamters used based on user config
    # the name "regressor__preprocess__encodeX__encoding_method" can be found using the above print statement
    param_grid = {
        'regressor__preprocess__encodeX__encoding_method': model_cfg['encoding'],
        'regressor__polyFeatures__degree': model_cfg['degree'],
    }

    # construct hyperparameter search and cross validation
    grid_pipeline = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=0, n_jobs=-1, error_score='raise')
    # fit the model
    grid_pipeline.fit(X_train, y_train)
    # store best hyperparamters
    best_param_dict = grid_pipeline.best_params_

    # Save the entire pipeline
    # This saves the pipeline for encoding, scaling and other transformation on input and output columns, as well as the best fit model
    joblib_file = "./model/linear_regression_pipeline.pkl"
    joblib.dump(grid_pipeline.best_estimator_, joblib_file)

    # Predict
    # The predicted result is scaled according to the transformation in the pipline. However, predict function for the estimator will internally call inverse() to unscale the results
    y_pred = grid_pipeline.predict(X_test)
    
    # Calculate R2
    r2 = r2_score(y_test, y_pred)

    print("\n\nfinish linear regression")
    print("Best parameters:", best_param_dict)
    print("R2 score:", r2)
    print(f"Took {time.time() - start_time} seconds")

    return r2, best_param_dict, time.time() - start_time
