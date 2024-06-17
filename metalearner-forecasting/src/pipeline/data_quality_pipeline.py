import os
import sys
import great_expectations as ge 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import logging
import yaml
from importlib import import_module
from src.exception import CustomException as e
from src.utils import read_process
import re
import pathlib
import seaborn as sns

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

class DataQualityCheck:

    def __init__(self,data):
        self.data = data

    def read_data(self, file_type=['excel','csv'], **kwargs):

        config = read_process()
        file_path = config.get('file_path', None)
        supported_file_types = config.get('supported_file_types', [])

        if file_path is None:
            raise ValueError("File path is not specified in the configuration.")

        if file_type not in supported_file_types:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: {', '.join(supported_file_types)}")

        try:
            if file_type == 'csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                data = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logging.info(f'Succesfully read data')
            #print(data.head())
            return data

        except Exception as e:
            logging.error(f"Error reading data from {file_type} file: {e}")
            return None

    def check_completeness(self, data):

        total_cells = data.size
        non_null_cells = data.count().sum()
        completeness_score = (non_null_cells / total_cells) * 100
        return completeness_score

    def check_duplicates(self, data):

        total_records = len(data)
        unique_records = len(set(data))
        duplicate_records = total_records - unique_records
        duplicate_percentage = (duplicate_records / total_records) * 100
        return duplicate_percentage
    
    def check_uniqueness(self, data):  
        try:
            total_rows = len(data)
            unique_data = data.drop_duplicates()
            unique_rows = unique_data.shape[0]
            uniqueness_percentage = (unique_rows / total_rows) * 100
            return uniqueness_percentage

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
        
    def check_dataset_size(self, data):
        try:
            num_rows = len(data)
            if num_rows < 1000:
                size_percentage = (num_rows / 500) * 100
                logging.info(f"The dataset has {num_rows} rows, which is {size_percentage:.2f}% of the minimum required size.")
                return size_percentage
            else:
                logging.info("The dataset has sufficient rows.")
                return 100.0  # indicating dataset is at least 500 rows
        except TypeError:
            logging.error("Input dataset is not iterable. Please provide a valid dataset.")
            return None
        
    def calculate_data_quality_score(self, completeness_score, duplicate_percentage, uniqueness_percentage, size_percentage):
        
        overall_data_quality_score = (completeness_score + duplicate_percentage + uniqueness_percentage + size_percentage) / 4
        return overall_data_quality_score