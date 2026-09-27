import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, TARGET

def preprocess_data(df):
    #select the features and target
    
    x = np.asarray(df[FEATURES])
    y = np.asarray(df[TARGET].astype(int))
    
    #standardise the features
    x_norm = StandardScaler().fit(x).transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_norm,
        y,
        test_size = 0.2,
        random_state = 42
    )
    
    return x_train, x_test, y_train, y_test


    