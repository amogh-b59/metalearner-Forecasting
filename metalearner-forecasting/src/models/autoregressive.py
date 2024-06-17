from constants.model import BaseModel
import numpy as np


class Model(BaseModel):
    def train(self, df, forecast):
        self.dataset = df[:-1*forecast,0]
        self.forecast = forecast
        self.output = []

    def __call__(self):
        for i in range(self.forecast):
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(self.dataset, lags=7, old_names=True).fit()
            yhat = model.forecast()
            self.dataset = np.append(self.dataset, yhat)
            self.output.append(yhat)
        self.output = np.array(self.output)
        self.output = self.output.reshape(1,self.output.shape[0])
        return self.output
    
model = Model()