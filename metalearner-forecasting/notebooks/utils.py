import os
import sys
from exception import CustomException as e
import yaml

# common functions for reusability


# 1. function for reading yaml files.
def read_process(file_path=r'D:\\Crider_Forecasting\\notebooks\\pre_process.yaml'):
    try:
        with open(file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading config file: {e}")
    
# 2. Function to read config.yaml file    
def read_config(file_path=r'D:\\Crider_Forecasting\\notebooks\\config.yaml'):
    try:
        with open(file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading config file: {e}")