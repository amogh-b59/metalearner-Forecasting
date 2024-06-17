import os
import sys
from src.exception import CustomException as e
import yaml

# 1. function for reading yaml files.
def read_process(file_path="D:/Amogh/Crider_Forecasting/crider_forecasting/config/pre_process.yaml"):
    try:
        with open(file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading config file: {e}")
    
# 2. Function to read config.yaml file    
def read_config(file_path="D:/Amogh/Crider_Forecasting/crider_forecasting/config/config.yaml"):
    try:
        with open(file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading config file: {e}")