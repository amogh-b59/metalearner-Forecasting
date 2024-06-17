import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from constants.model import BaseModel


class Model(BaseModel):
    def train(self, df, forecast):
        self.X_train, self.y_train, self.predict = self._shape_data(df=df, lag=self.lag, forecast=forecast)

        model = Sequential()
        model.add(LSTM(units = 500, return_sequences = True, input_shape=self.X_train.shape[1:]))
        model.add(Dropout(0.1))
        model.add(LSTM(units = 500, return_sequences = True))
        model.add(Dropout(0.1))
        model.add(LSTM(units = 500))
        model.add(Dropout(0.1))
        model.add(Dense(units=forecast))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.model = model
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=20, verbose=0)

    def __call__(self):
        prices = self.model.predict(self.predict)
        return prices

model = Model(epochs=500, lag=3)