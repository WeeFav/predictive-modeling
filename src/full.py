import pandas as pd
import numpy as np
import time
import yaml
import warnings
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from knn import knn
from decision_tree import decision_tree
from random_forest import random_forest
from gradient_boosting import gradient_boosting
from neural_network import neural_network
import json
import shutil
import os


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start_time = time.time()

    # create model directory to store current run results
    try:
        if os.path.exists('model'):
            shutil.rmtree('model')
        os.makedirs('model')
    except OSError as e:
        pass

    # read in user configurations of the dataset
    with open('config/preprocessing.yaml', 'rb') as file:
        preprocess_cfg = yaml.load(file, Loader=yaml.FullLoader)
    
    # preprocess the data
    # the returned categorical_columns, numerical_columns, and binary_columns DOES NOT include output column
    df, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type = preprocess_data(preprocess_cfg)

    # logging config used
    config = {
        'filepath': preprocess_cfg['filepath'],
        'input_columns': preprocess_cfg['input_columns'],
        'output_column': preprocess_cfg['output_column'],
        'date_columns': preprocess_cfg['date_columns'],
        'numerical_columns': numerical_columns, 
        'categorical_columns': categorical_columns, 
        'binary_columns': binary_columns, 
        'output_column_name': output_column_name,
        'output_column_type': output_column_type
    }

    config_path = './model/config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print('numerical_columns: ', numerical_columns)
    print('categorical_columns: ', categorical_columns)
    print('binary_columns: ', binary_columns)

    # input columns
    X = df.drop(columns=output_column_name)
    # output column
    y = df[output_column_name].to_frame()
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    print(X.shape)
    print(y.shape)

    # write testing data to directory
    X_test.to_csv('./model/X_test.csv', index=False)
    y_test.to_csv('./model/y_test.csv', index=False)

    score_list = []
    algorithm_list = []
    result_list = []

    ### linear_regression, logistic_regression, knn ###
    if output_column_type == 'numerical':
        score, param_dict, processing_time = linear_regression(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
        score_list.append(score)
        algorithm_list.append('linear regression')
        result_list.append({
            'algorithm': 'linear regression',
            'score': score,
            'param_dict': param_dict,
            'processing_time': processing_time
        })
    elif output_column_type == 'binary':
        score, param_dict, processing_time = logistic_regression(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
        score_list.append(score)
        algorithm_list.append('logistic regression')
        result_list.append({
            'algorithm': 'logistic regression',
            'score': score,
            'param_dict': param_dict,
            'processing_time': processing_time
        })
    elif output_column_type == 'categorical':
        score, param_dict, processing_time = knn(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
        score_list.append(score)
        algorithm_list.append('knn')
        result_list.append({
            'algorithm': 'knn',
            'score': score,
            'param_dict': param_dict,
            'processing_time': processing_time
        })

    ### decision_tree ###
    score, param_dict, processing_time = decision_tree(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
    score_list.append(score)
    algorithm_list.append('decision tree')
    result_list.append({
        'algorithm': 'decision tree',
        'score': score,
        'param_dict': param_dict,
        'processing_time': processing_time
    })

    ### random_forest ###
    score, param_dict, processing_time = random_forest(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
    score_list.append(score)
    algorithm_list.append('random forest')
    result_list.append({
        'algorithm': 'random forest',
        'score': score,
        'param_dict': param_dict,
        'processing_time': processing_time
    })

    ### gradient_boosting ###
    score, param_dict, processing_time = gradient_boosting(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
    score_list.append(score)
    algorithm_list.append('gradient boosting')
    result_list.append({
        'algorithm': 'gradient boosting',
        'score': score,
        'param_dict': param_dict,
        'processing_time': processing_time
    })

    ### neural_network ###
    score, param_dict, processing_time = neural_network(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type)
    score_list.append(score)
    algorithm_list.append('neural network')
    result_list.append({
        'algorithm': 'neural network',
        'score': score,
        'param_dict': param_dict,
        'processing_time': processing_time
    })

    with open('./model/result.json', 'w') as f:
        json.dump(result_list, f, indent=4)

    print("\n\n------------------------------------------result-----------------------------------------------")
    idx = np.argmax(score_list)
    print("best model: ", algorithm_list[idx])
    if output_column_type == 'numerical':
        print("r2 score: ", score_list[idx])
    else:
        print("accuracy: ", score_list[idx])
    print("--- %s seconds total ---" % (time.time() - start_time))

