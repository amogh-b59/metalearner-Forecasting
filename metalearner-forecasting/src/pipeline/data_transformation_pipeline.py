import os
from importlib import import_module
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

class DataTransformationPipeline:

    def __init__(self, preprocessed_data, window=55, forecast=1, shift=7, single=False):
        self._data = preprocessed_data.copy()
        self.window = window
        self.forecast = forecast
        self.shift = shift
        logging.info("Done Initializing")

        if single is False:
            self.rolling_transformation()
            logging.info("called //rolling_transformation()// method")
    

    def rolling_transformation(self): #for LSTM
        num_windows = int((len(self._data) - self.window) / self.shift) + 1
        dataset = []
        for i in range(num_windows):
            min = 0 + self.shift * i 
            max = self.window + self.shift * i 
            array = self._data[min:max] 
            np_array = np.array(array) 
            dataset.append(np_array) 
        self.dataset = np.array(dataset)
        one = self.dataset[:,:,1:].copy()
        input_1 = []
        for i in range(one.shape[0]):
            input_1.append( one[i] )
        self.input_1 = np.array(input_1)
        two = self.dataset[:,:-1*self.forecast,0:1].copy()
        input_2 = []
        for i in range(two.shape[0]):
            input_2.append( two[i] )
        self.input_2 = np.array(input_2)
        out = self.dataset[:,-1*self.forecast:,0:1].copy()
        output = []
        for i in range(out.shape[0]):
            output.append(  out[i]  )
        self.output = np.array(output)


    def build_models(self):
        from src import models as model_location
        abs_path = os.path.dirname(model_location.__file__)
        foo = os.listdir(abs_path)
        models = []
        for file in foo:
            if file[0] != '_' and file[0] != '.':
                module_name = file.split(".")[0]
                module = import_module("src.models."+module_name)
                model = getattr(module, 'model')
                models.append(model)
        self.models = models

        
    def build_training_data(self):
        predictions = []
        count=0
        for batch in self.dataset.copy(): 
            count += 1
            logging.info("Batch " + str(count))
            batch_predictions = []
            for model in self.models:
                model.train(batch.copy(), self.forecast)
                pred = model()
                batch_predictions.append(pred)
            batch_predictions = np.array(batch_predictions)
            predictions.append(batch_predictions)
        logging.info("Done training models")
        self.predictions = np.array(predictions)
        self.predictions = self.predictions.reshape(self.predictions.shape[0], self.predictions.shape[1], self.predictions.shape[-1])
        three = self.predictions
        input_3 = []
        for i in range(three.shape[0]):
            input_3.append(  three[i]  )
        self.input_3 = np.array(input_3)
        
        
    def inverse_transform(self, data):
        return self.mlsc.inverse_transform(data)