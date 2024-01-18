"""
Data Preprocessing Module

This module loads the facial features dataset from the database into X(features) and Y(targets)
variables and performs one hot encoding of target values.
"""

from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import sys
import os

def preprocessing(path, logger):
    '''
        Loads the dataset from database and performs one-hot encoding
        of target values(names)

        params:
            path: path of current working directory
            logger: Logging module
    '''

    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = 'attendance_system'
    COLLECTION_NAME = 'FB_DB'
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    try:
        # MongoDB localhost connection
        client = MongoClient(DEFAULT_CONNECTION_URL) 
        db = client[DB_NAME]
        #Collection of facial features dataset
        data = db[COLLECTION_NAME]

        if db:
            print("Connection established successfully")
        else:
            print("Connection to db failed")

        #Loading features from database to X and Y
        df = pd.DataFrame(list(data.find()))
        # Removing first column as it contains _id values
        df.drop('_id', axis=1, inplace=True)
        X = df.iloc[:, :-1]
        print("X values loaded successfully from database")
        Y = df.iloc[:, -1]
        print("Y values loaded successfully from database")

        # Performing label encoding
        Y_encoded = label_encoder.fit_transform(Y)
        print("Label encoding of names completed")

        #Saving label encoder
        with open(os.path.join(path, "label_encoder.pkl") , "wb") as f:
            pickle.dump(label_encoder, f )
        print("Label encoder object saved")
        
        Y_encoded = Y_encoded.reshape(-1,1)
        # Performing one hot encoding
        Y_onehot = one_hot_encoder.fit_transform(Y_encoded).toarray()
        print("One hot encoding of names completed")

    except Exception as e:
        print("Error at module: Data Preprocessing")
        print("Error at line number: ", sys.exc_info()[-1].tb_lineno)
        print("Error:",e)
        #Add logger part

    