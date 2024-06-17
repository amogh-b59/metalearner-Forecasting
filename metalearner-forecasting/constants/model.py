from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    def __init__(self, epochs=None, lag=None):
        self.epochs=epochs
        self.lag = lag
    
    def _shape_data(self, df, lag, forecast):
        X_train = []
        y_train = []
        X = []
        for i in range(lag, len(df)-forecast+1):
            if i < (len(df)-forecast):
                time_features = df[(i-lag):i, 0]
                one_hot_features = df[i, 1:]
                features = np.concatenate((time_features, one_hot_features))
                X_train.append(features)
                y_train.append(df[i:i+forecast, 0])
            else:
                time_features = time_features = df[(i-lag):i, 0]
                one_hot_features = df[i, 1:]
                X.append(features)

        X_train, y_train, X = np.array(X_train), np.array(y_train), np.array(X)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X_train, y_train, X
    
    def _shape_training_data(self, df, lag, forecast):
        X_train = []
        y_train = []
        X = []
        for i in range(lag, len(df)-forecast+1):
            if i < (len(df)-forecast):
                time_features = df[(i-lag):i, 0]
                one_hot_features = df[i, 1:]
                features = np.concatenate((time_features, one_hot_features))
                X_train.append(features)
                y_train.append(df[i:i+forecast, 0])
            else:
                time_features = time_features = df[(i-lag):i, 0]
                one_hot_features = df[i, 1:]
                X.append(features)

        X_train, y_train, X = np.array(X_train), np.array(y_train), np.array(X)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X_train, y_train, X
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def __call__(self):
        pass
