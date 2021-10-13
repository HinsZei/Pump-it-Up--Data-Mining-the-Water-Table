import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

'''Converting labels to 0,1'''


def label_pipe():
    label_pipeline = Pipeline([
        ('label_encoder', OneHotEncoder(drop='if_binary', dtype=int))
    ])
    return label_pipeline


'''Filling numerical data with the median and then standardising it'''


def numerical_pipe():
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())])
    return num_pipeline


'''Filling with plural for categorical data and then one-hot encoding'''


def categorical_pipe():
    cat_pipeline = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(drop='if_binary', dtype=int))
    ])
    return cat_pipeline


'''Construct a transformer that applies the numerical pipeline to the numeric columns of a dataframe'''


def num_transformer():
    num_transfer = ColumnTransformer([
        ('num', numerical_pipe(), selector(dtype_include='number'))])
    return num_transfer


'''
Construct a transformer that applies the numerical pipeline to the numeric columns and 
applies the categorical pipeline to categorical columns
'''


def all_transformer():
    all_transformer = ColumnTransformer([
        ('num', numerical_pipe(), selector(dtype_include='number')),
        ('categories', categorical_pipe(), selector(dtype_include='category'))
    ])
    return all_transformer


def numerical_data(path, isPredict):
    data = pd.read_csv(path)
    data.set_index(['id'], inplace=True)  # set id as index for getting it in predict()
    # all numerical columns
    X = data.loc[:, ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'population']]
    if not isPredict:
        # if the dataset have labels then transform it
        y = data.loc[:, ['status_group']]
        label_transfer = ColumnTransformer([
            ('label', label_pipe(), ['status_group'])
        ])
        y = np.ravel(label_transfer.fit_transform(y))
        return X, y
    else:
        return X


def all_data(path, isPredict):
    data = pd.read_csv(path)
    data.set_index(['id'], inplace=True)
    # all the numerical columns and some categorical columns that can be used
    X = data[['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'population', 'public_meeting',
              'permit', 'extraction_type_class', 'management_group', 'payment_type',
              'quality_group',
              'quantity_group', 'source_class', 'waterpoint_type_group']]
    if not isPredict:
        y = data.loc[:, ['status_group']]
        label_transfer = ColumnTransformer([
            ('label', label_pipe(), ['status_group'])
        ])
        y = np.ravel(label_transfer.fit_transform(y))
        return X, y
    else:
        return X


def multiple_class_data(path, isPredict):
    data = pd.read_csv(path)
    data.set_index(['id'], inplace=True)
    # this part is similar to previous method
    X = data[['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'population', 'public_meeting',
              'permit', 'extraction_type_class', 'management_group', 'payment_type',
              'quality_group',
              'quantity_group', 'source_class', 'waterpoint_type_group']]
    if not isPredict:
        # As it is a multiclass dataset, use Label encoder instead of one hot encoder
        y = np.ravel(data.loc[:, ['status_group']])
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        return X, y
    else:
        return X


''' the following methods are simply implemented resampling methods, their performance is similar even SmoteTomek did the best'''


def oversampling_Smote():
    sampler = SMOTE(random_state=0)
    return sampler


def oversampling_SmoteTomek():
    sampler = SMOTETomek(random_state=0)
    return sampler


def oversampling_ADASYN():
    sampler = ADASYN(random_state=0)
    return sampler


def undersampling_TomekLinks():
    sampler = TomekLinks()
    return sampler


'''abandoned'''


def features_selection():
    selector = SelectFromModel(ExtraTreesClassifier(), prefit=False)
    return selector
