import pandas as pd
import numpy as np
from preprocessing import EncodeX
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import keras
from keras import layers
import yaml
import time
from sklearn.metrics import r2_score, accuracy_score
import joblib

def neural_network(X_train, X_test, y_train, y_test, categorical_columns, numerical_columns, binary_columns, output_column_name, output_column_type):
    start_time = time.time()

    # get hyperparameters
    with open('config/neural_network.yaml', 'rb') as file:
        model_cfg = yaml.load(file, Loader=yaml.FullLoader)

    p = len(X_train.columns)
    n = len(y_test)

    # encode categorical data
    if len(categorical_columns) > 0:
        encoderX = EncodeX("ordinal")
        encoderX.fit(X_train[categorical_columns], y_train)
        X_train[categorical_columns] = encoderX.transform(X_train[categorical_columns])
        X_test[categorical_columns] = encoderX.transform(X_test[categorical_columns])
        # save encoderX
        joblib_file = "./model/encoderX.pkl"
        joblib.dump(encoderX, joblib_file)
    if output_column_type == 'categorical':
        # y can only choose 'ordinal' as encoding method
        encoderY = EncodeX("ordinal")
        encoderY.fit(y_train)
        y_train = encoderY.transform(y_train)
        y_test = encoderY.transform(y_test)
        # save encoderY
        joblib_file = "./model/encoderY.pkl"
        joblib.dump(encoderY, joblib_file)

    # normalize numeric data
    if len(numerical_columns) > 0:
        scalerX = MinMaxScaler()  
        scalerX.fit(X_train[numerical_columns])
        X_train[numerical_columns] = scalerX.transform(X_train[numerical_columns])
        X_test[numerical_columns] = scalerX.transform(X_test[numerical_columns])
        # save scalerX
        joblib_file = "./model/scalerX.pkl"
        joblib.dump(scalerX, joblib_file)
    if output_column_type == 'numerical':
        scalerY = MinMaxScaler()  
        scalerY.fit(y_train)
        y_train = scalerY.transform(y_train)
        y_test = scalerY.transform(y_test)
        # save scalerY
        joblib_file = "./model/scalerY.pkl"
        joblib.dump(scalerY, joblib_file)

    # print("X_train shape: ", X_train.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_train shape: ", y_train.shape)
    # print("y_test shape: ", y_test.shape)
    # print(X_train)
    # print(y_train)

    # Build model
    input_list = []
    # build embedding layers for categorical columns
    emb_list = []
    for i, col in enumerate(categorical_columns):
        # choose embedding dimension
        if (model_cfg['embedding_dimension'] == 'formula'):
            max_int = X_train[col].max() + 1    
            # print(col, max_int - 1)
            embedding_dim = round((max_int-1)**0.25)
        else:
            embedding_dim = model_cfg['embedding_dimension']

        # for each column, there will be a input layer, then followed by embedding layer
        inp = layers.Input(shape=(1,))
        input_list.append(inp)
        emb = layers.Embedding(input_dim=max_int, output_dim=embedding_dim)(inp)
        emb_list.append(emb)

    # input layer for numeric and binary columns
    numerical_dim = len(numerical_columns + binary_columns)
    # input is of numerical_columns + binary_columns features
    numerical_input = layers.Input(shape=(numerical_dim,))
    input_list.append(numerical_input)
    # need to reshape input to be the same shape as embedding layers if embedding layers are used
    if (len(categorical_columns) > 0):
        numerical_reshape = layers.Reshape((1, numerical_dim))(numerical_input)
    else:
        numerical_reshape = numerical_input
    emb_list.append(numerical_reshape)

    # configure output
    if output_column_type == 'numerical': 
        out_dim = 1
        activation = None
        loss_fn = keras.losses.mean_squared_error
        metrics = [keras.metrics.mean_absolute_error]
    elif output_column_type == 'categorical':
        out_dim = len(y_train[output_column_name].unique()) + 1 # add 1 for unknown category
        y_train = keras.utils.to_categorical(y_train, num_classes=out_dim)
        activation = 'softmax'
        loss_fn = keras.losses.categorical_crossentropy
        metrics = ['accuracy']
    else:
        out_dim = 1
        activation = 'sigmoid'
        loss_fn = keras.losses.binary_crossentropy
        metrics = ['accuracy']

    # hidden and output layer
    joinedInput = layers.Concatenate()(emb_list) # concatnate input across all columns
    reshape = layers.Reshape((-1,))(joinedInput)
    dense1 = layers.Dense(model_cfg['hidden_layer'], activation=model_cfg['activation'])(reshape)
    dense2 = layers.Dense(model_cfg['hidden_layer'], activation=model_cfg['activation'])(dense1)
    out = layers.Dense(out_dim, activation=activation)(dense2)
    model = keras.Model(inputs=input_list, outputs=out, name="Predict")
    # model.summary()
    # utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(loss=loss_fn, optimizer=model_cfg['optimizer'], metrics=metrics)

    # prepare inputs
    input_df_list = [X_train[col] for col in categorical_columns]
    numerical_df = X_train.drop(categorical_columns, axis=1)
    input_df_list.append(numerical_df)
    # input_df_list will be [df for categoic column 1, df for categoic column 2, ..., df for all numeric and binary columns]

    # save model
    model_save_path = "./model/neural_network_model.keras"
    checkpoint_path = "./ckpt/checkpoint.weights.h5"
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_best_only=False,
                                                save_weights_only=True)

    model.fit(x=input_df_list,
              y=y_train,
              epochs=model_cfg['epochs'],
              batch_size=model_cfg['batch_size'],
              callbacks=[cp_callback],
              verbose=0)

    # Save the entire model
    model.save(model_save_path)

    # Predict
    model.load_weights(checkpoint_path)
    # prepare inputs
    input_df_list = [X_test[col] for col in categorical_columns]
    numerical_df = X_test.drop(categorical_columns, axis=1)
    input_df_list.append(numerical_df)
    y_pred = model.predict(input_df_list, batch_size=model_cfg['batch_size'], verbose=0)

    # Print out prediction results and evaluate
    if output_column_type == 'numerical':
        # print('Mean Absolute Error (normalized):', sklearn.metrics.mean_absolute_error(y_test.to_numpy(), y_pred.squeeze()))
        y_pred = scalerY.inverse_transform(y_pred)
        y_test_unscaled = scalerY.inverse_transform(y_test)
        r2 = r2_score(y_test_unscaled, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        score = adj_r2
        # print(y_test, y_pred)
        # print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test.squeeze(), y_pred.squeeze()))
    elif output_column_type == 'categorical':
        # print(f'Actual: {y_test.to_numpy().squeeze()[0:9]}, Predicted: {np.argmax(y_pred, axis=1)[0:9]}')
        acc = accuracy_score(y_test.to_numpy().squeeze(), np.argmax(y_pred, axis=1))
        score = acc
        # print('Accuracy:', acc)
    elif output_column_type == 'binary':
        y_test = y_test.to_numpy().squeeze()
        y_pred = np.round(y_pred.squeeze())
        # print(f'Actual: {y_test[0:9]}, Predicted: {y_pred[0:9]}')
        acc = accuracy_score(y_test, y_pred)
        score = acc
        # print('Accuracy:', acc)

    param_dict = {
        'encoding': model_cfg['encoding'],
        'embedding_dimension': model_cfg['embedding_dimension'],
        'optimizer': model_cfg['optimizer'],
        'activation': model_cfg['activation'],
        'hidden_layer': model_cfg['hidden_layer'],
        'batch_size': model_cfg['batch_size'],
        'epochs': model_cfg['epochs']
    }

    print("\n\nfinish neural network")
    print("best param:", param_dict)
    if output_column_type == 'numerical':  
        print("r2 score:", score)
    else:
        print("accuracy:", score)
    print(f"took {time.time() - start_time} seconds")

    return score, param_dict, time.time() - start_time
