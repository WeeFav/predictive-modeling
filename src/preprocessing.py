import pandas as pd
import numpy as np
from dateutil.parser import parse
import category_encoders as ce
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher 
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin

# In order to use this class in ColumnTransformer, we need to inherit from BaseEstimator and TransformerMixin and must contain fit and transform method
class EncodeX(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_method='ordinal'):
        self.encoding_method = encoding_method
        if self.encoding_method == 'ordinal':
            self.encoder = ce.OrdinalEncoder(handle_unknown='value')
        elif self.encoding_method == 'target':
            self.encoder = ce.TargetEncoder(handle_unknown='value')
        else:
            assert False, f"{encoding_method} not supported" 

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        X = self.encoder.transform(X, y)
        return X

class one_hot_encoding():
    def __init__(self, categorical_columns):
        self.encoder = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')
        self.categorical_columns = categorical_columns

    def encode_train(self, X_train):
        selected_columns = X_train[self.categorical_columns] # first filter out columns because sklearn doesn't allow us to choose which columns get encoded
        date_columns = X_train.drop(self.categorical_columns, axis=1)
        self.encoder.fit(selected_columns)
        selected_columns = self.encoder.transform(selected_columns).toarray() # sklearn only outputs array
        return np.concatenate([selected_columns, date_columns], axis=1)
    
    def encode_test(self, X_test):
        selected_columns = X_test[self.categorical_columns] # first filter out columns because sklearn doesn't allow us to choose which columns get encoded
        date_columns = X_test.drop(self.categorical_columns, axis=1)
        # no fitting
        selected_columns = self.encoder.transform(selected_columns).toarray() # transform based on trained data
        return np.concatenate([selected_columns, date_columns], axis=1)       

# label encoding from sklearn doesn't deal with unseen data during training and can't apply fit/transform across multiple columns
# use ordianl encoding from category_encoders instead
class ordinal_encoding():
    def __init__(self, categorical_columns):
        self.encoder = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='value')
    
    def encode_train(self, X_train):
        self.encoder.fit(X_train)
        X_train = self.encoder.transform(X_train)
        return X_train
    
    def encode_test(self, X_test):
        X_test = self.encoder.transform(X_test)
        return X_test        


class target_encoding():
    def __init__(self, categorical_columns):
        self.encoder = ce.TargetEncoder(cols=categorical_columns, handle_unknown='value')
    
    def encode_train(self, X_train, y_train):
        self.encoder.fit(X_train, y_train)
        X_train = self.encoder.transform(X_train)
        return X_train
    
    def encode_test(self, X_test):
        X_test = self.encoder.transform(X_test)
        return X_test
    

class binary_encoding():
    def __init__(self, categorical_columns):
        self.encoder = ce.BinaryEncoder(cols=categorical_columns, handle_unknown='value')
    
    def encode_train(self, X_train):
        self.encoder.fit(X_train)
        X_train = self.encoder.transform(X_train)
        return X_train
    
    def encode_test(self, X_test):
        X_test = self.encoder.transform(X_test)
        return X_test

# variant of target encoding
class catboost_encoding():
    def __init__(self, categorical_columns):
        self.encoder = ce.CatBoostEncoder(cols=categorical_columns, handle_unknown='value')
    
    def encode_train(self, X_train, y_train):
        self.encoder.fit(X_train, y_train)
        X_train = self.encoder.transform(X_train)
        return X_train
    
    def encode_test(self, X_test):
        X_test = self.encoder.transform(X_test)
        return X_test

class hash_encoding():
    def __init__(self, categorical_columns):
        self.encoder = FeatureHasher(n_features=20, input_type="string")
        self.categorical_columns = categorical_columns
    
    def encode_train(self, X_train):
        selected_columns = X_train[self.categorical_columns] # first filter out columns because sklearn doesn't allow us to choose which columns get encoded
        date_columns = X_train.drop(self.categorical_columns, axis=1)
        selected_columns = selected_columns.to_numpy().astype(str) # force every element in numpy array to be string
        selected_columns = self.encoder.transform(selected_columns).toarray()
        return np.concatenate([selected_columns, date_columns], axis=1)
    
    def encode_test(self, X_test):
        selected_columns = X_test[self.categorical_columns] # first filter out columns because sklearn doesn't allow us to choose which columns get encoded
        date_columns = X_test.drop(self.categorical_columns, axis=1)
        selected_columns = selected_columns.to_numpy().astype(str) # force every element in numpy array to be string
        selected_columns = self.encoder.transform(selected_columns).toarray()
        return np.concatenate([selected_columns, date_columns], axis=1)
    
# variant of target encoding
class loo_encoding():
    def __init__(self, categorical_columns):
        self.encoder = ce.LeaveOneOutEncoder(cols=categorical_columns, handle_unknown='value')
    
    def encode_train(self, X_train, y_train):
        self.encoder.fit(X_train, y_train)
        X_train = self.encoder.transform(X_train)
        return X_train
    
    def encode_test(self, X_test):
        X_test = self.encoder.transform(X_test)
        return X_test

class normalize_data():
    def __init__(self, numerical_columns):
        self.scaler = preprocessing.MinMaxScaler()
        self.numerical_columns = numerical_columns
    
    def scaler_train(self, X_train):
        X_train.loc[:, self.numerical_columns] = self.scaler.fit_transform(pd.DataFrame(X_train[self.numerical_columns]))
        return X_train
    
    def scaler_test(self, X_test):
        X_test.loc[:, self.numerical_columns] = self.scaler.transform(pd.DataFrame(X_test[self.numerical_columns]))
        return X_test    


def preprocess_data(cfg, df_data = None):
    """
    used for determine column type, transform date columns into numerical columns, drop columns and fill in missing data, and convert binary data to 0 and 1
    """
    df = pd.read_csv(os.path.join("datasets", cfg['filepath'])) if df_data is None else df_data

    # determine column type
    input_columns = cfg['input_columns'].copy()
    output_column = cfg['output_column']
    used_columns = input_columns + [output_column] # since output_column is just a string, turn in into a list to concatnate

    # determine if a column is binary by checking if the data value only contains [0, 1, 'T', ...]
    binary_columns = [col for col in df[used_columns] if np.isin(df[col].dropna().unique(), [0, 1, 'T', 'F', 'Y', 'N', 'True', 'true', 'False', 'false', 'Yes', 'yes', 'No', 'no']).all()]
    date_columns = cfg['date_columns']

    assert output_column not in date_columns, "output cannot be date"

    # determine whether a column is numeric or categoric
    # first exclude binary and date column
    non_date_binary_columns = [x for x in used_columns if x not in (date_columns + binary_columns)]
    dtypes_dict = df[non_date_binary_columns].dtypes.to_dict()
    categorical_columns = []
    numerical_columns = []
    # then loop through rest of the columns to determine numeric or categoric
    for col_name, typ in dtypes_dict.items():
        if (typ == 'O'):
            categorical_columns.append(col_name)
        else:
            numerical_columns.append(col_name)

    # transform date columns into numerical columns
    # date columns are useless after this
    for col in date_columns:
        df[col].fillna("1/01/0001", inplace=True) # first fill up missing values
        df[col] = df[col].map(lambda x: parse(x).strftime('%d-%m-%Y')) # convert into standard format
        df[[f'{col}_day', f'{col}_month', f'{col}_year']] = df[col].str.split('-', expand=True) # split into day, month, year
        df[f'{col}_year'] = df[f'{col}_year'].map(lambda x: x[-2:]) # convert 2023 -> 23 # THIS MIGHT NOT BE A GOOD IDEA
        df.drop(columns=col, axis=1, inplace=True) # drop original column since we already have new ones
        # convert these columns from string to numbers
        df[f'{col}_day'] = pd.to_numeric(df[f'{col}_day'])
        df[f'{col}_month'] = pd.to_numeric(df[f'{col}_month'])
        df[f'{col}_year'] = pd.to_numeric(df[f'{col}_year'])
        numerical_columns.append(f'{col}_day')
        numerical_columns.append(f'{col}_month')
        numerical_columns.append(f'{col}_year')
        input_columns.append(f'{col}_day')
        input_columns.append(f'{col}_month')
        input_columns.append(f'{col}_year')
        input_columns.remove(col)
    
    used_columns = input_columns + [output_column]
    assert len(used_columns) == len(numerical_columns + categorical_columns + binary_columns)

    # construct table for visualization
    input_output = []
    column_type = []
    for col in df.columns:
        if col in input_columns:
            input_output.append("input")
        elif col == output_column:
            input_output.append("output")
        else:
            input_output.append("n/a")
        
        if col in categorical_columns:
            column_type.append("categorical")
        elif col in numerical_columns:
            column_type.append("numeric")
        elif col in binary_columns:
            column_type.append("binary")
        else:
            column_type.append("n/a")

    column_type_table = {
        "column name" : df.columns,
        "input/output" : input_output,
        "type" : column_type
    }
    print(pd.DataFrame(column_type_table))




    # drop not used columns
    df = pd.DataFrame(df, columns=used_columns)
    # drop rows that are N/A in output column
    df.dropna(subset=output_column, inplace=True)
    assert(df[output_column].isnull().values.any() == False)

    # fill in different values for N/A in input column according to column type
    for col in numerical_columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in binary_columns:
        df[col].fillna(df[col].mode(), inplace=True)
    df.fillna('not available', inplace=True)
    df.replace(r'\?+', 'not available', regex=True, inplace=True)

    # map binary columns to values 0 and 1
    for col in binary_columns:
        df[col].replace([0, 'F', 'N', 'False', 'false', 'No', 'no'], 0, inplace=True)
        df[col].replace([1, 'T', 'Y', 'True', 'true', 'Yes', 'yes'], 1, inplace=True)

    # store output column type
    if output_column in categorical_columns:
        output_column_type = 'categorical'
        categorical_columns.remove(output_column)
    elif output_column in numerical_columns:
        output_column_type = 'numerical'
        numerical_columns.remove(output_column)
    else:
        output_column_type = 'binary'
        binary_columns.remove(output_column)
    
    assert len(categorical_columns + numerical_columns + binary_columns) + 1 == len(df.columns) # +1 because output column is not included

    # the returned categorical_columns, numerical_columns, and binary_columns DOES NOT include output column
    return df, categorical_columns, numerical_columns, binary_columns, output_column, output_column_type

# encode catergorical features for train and test data
def encode_train_test(X_train, X_test, y_train, encoding_method, categorical_columns):
    if (encoding_method == "one_hot"):
        encoder = one_hot_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "ordinal"):
        # integers are selected at random
        encoder = ordinal_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "target"):
        encoder = target_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train, y_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "binary"):
        encoder = binary_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "catboost"):
        encoder = catboost_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train, y_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "hash"):
        encoder = hash_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "loo"):
        encoder = loo_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train, y_train)
        X_test = encoder.encode_test(X_test)
    elif (encoding_method == "embedding"):
        # embedding layers need to turn words into numbers first
        encoder = ordinal_encoding(categorical_columns)
        X_train = encoder.encode_train(X_train)
        X_test = encoder.encode_test(X_test)
    else:
        print("Invalid encoding method")
        sys.exit()

    return X_train, X_test

