import os
import sys
import logging
import numpy as np
import yaml

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

from src.utils import read_process
from src.pipeline.data_preprocess_pipeline import DataPreprocessPipeline

def read_data(file_type=['excel'], **kwargs):
    preprocess_pipeline = DataPreprocessPipeline(None) 
    data = preprocess_pipeline.read_data(file_type=file_type, **kwargs)
    return data

def main():
    data = read_data(file_type='excel')
    preprocess_pipeline = DataPreprocessPipeline(data)

    # List of functions to execute in sequence
    preprocess_functions = [
        preprocess_pipeline.check_leading_spaces,
        preprocess_pipeline.detect_date_column,
    ]

    for preprocess_function in preprocess_functions:
            preprocess_function(data)

    date_columns = preprocess_pipeline.check_date_columns_keyword(data)
    data = preprocess_pipeline.remove_prefixes(data, date_columns)
    data = preprocess_pipeline.sort_dataset_by_date(data, date_columns)
    data = preprocess_pipeline.extract_date_part(data,date_columns)
    data = preprocess_pipeline.set_date_column_as_index(data, date_columns[0])
    numerical_columns, categorical_columns = preprocess_pipeline.detect_column_types(data)
    data = preprocess_pipeline.convert_numerical_columns_to_float(data, numerical_columns)
    target_columns = preprocess_pipeline.find_target_column(data) 
    #data = preprocess_pipeline.remove_special_characters(data, target_columns)
    data = preprocess_pipeline.handle_null_values(data)
    outliers = preprocess_pipeline.detect_outliers(data, target_columns)
    data = preprocess_pipeline.winsorize_data(data, target_columns, lower_percentile=0.05, upper_percentile=0.95)
    data = preprocess_pipeline.one_hot_encode_categorical_columns(data, categorical_columns)
    selected_data = preprocess_pipeline.correlation_analysis(data, target_columns, threshold=0.5)
    normalised_data = preprocess_pipeline.normalize_dataset(selected_data)
    normalised_data = preprocess_pipeline.move_target_to_first(normalised_data, target_columns)
    np_data = preprocess_pipeline.convert_to_numpy(normalised_data)
    processed_data = preprocess_pipeline.save_np_data_to_processed_dir(np_data)
    print(np_data)

if __name__ == "__main__":
    main()