import os
import sys
from importlib import import_module
import numpy as np
import pandas as pd
from src.exception import CustomException as e
import yaml
import logging
from src.utils import read_process
import re
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

class DataPreprocessPipeline:

        def __init__(self,data):
            self.data = data

        def read_data(self, file_type='excel', **kwargs):
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
            
        def check_leading_spaces(self, data):
        
            try:
                columns_with_spaces = [col for col in data.columns if col.startswith(" ")]
                logging.info('check_leading_spaces completed')
                # print(data.head())
                return columns_with_spaces
            except Exception as e:
                logging.error(f'error checking for leading spaces: {e}')
                #print(data.head())
                return None

        def detect_date_column(self, data):
        
            config=read_process()
            date_formats = config.get('date_formats', [])

            if not date_formats:
                logging.warning("No date formats specified in the config file. Using default formats.")
            for column in data.columns:
                for date_format in date_formats:
                    try:
                        pd.to_datetime(data[column], format=date_format, errors='coerce')
                        logging.info(f"Detected '{column}' as a potential date column using format '{date_format}'.")
                        
                        return column
                    except ValueError:
                        pass  # Ignore errors and try the next format
            logging.info("No potential date column detected.")
            return None
        
        def check_date_columns_keyword(self, data):

            config = read_process()
            date_column_keywords = config.get('date_column_keywords', [])
            detected_date_columns=[]
            date_columns = [col for col in data.columns if any(keyword.lower() in col.lower() for keyword in date_column_keywords)]
            if date_columns:
                logging.info("Using 'date_columns' parameter from config file, potential date column found" + str(date_columns))
                for col in date_columns:
                    # print(f"- {col}")
                    detected_date_columns.append(col)
            else:
                logging.error("No date columns found")
            return date_columns
        
        def sort_dataset_by_date(self, data, date_columns):
            try:
                if date_columns:
                    if any(col in data.columns for col in date_columns):
                        sorted_data = data.sort_values(by=date_columns)
                        logging.info("Data Sorted")
                        #print(data.head())
                        return sorted_data
                    else:
                        logging.warning(f"None of the specified date columns {date_columns} found in the dataset. Sorting by index instead.")
                        sorted_data = data.sort_index()
                        return sorted_data
                else:
                    logging.warning("No date columns provided for sorting. Sorting by index instead.")
                    sorted_data = data.sort_index()
                    #print(sorted_data.head())
                    return sorted_data
            except Exception as e:
                logging.error(f"Error occurred while sorting dataset by date: {str(e)}")
                return None
            
        def remove_prefixes(self, data, date_columns):
            for col in date_columns:
                try:
                    for index, value in data[col].items():
                        if re.match(r'^\d+\s+w/e\s+', str(value)):
                            data.at[index, col] = re.sub(r'^\d+\s+w/e\s+', '', str(value))
                            
                    logging.info("Prefix removed from date column")
                    #print(data.head())
                except Exception as e:
                    logging.error(f"An error occurred while processing column {col}: {e}")
            return data

        def extract_date_part(self, data, date_columns):
                
                try:
                    for col in date_columns:
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                        data[col] = data[col].dt.strftime('%m-%d-%y')
                    logging.info("Date parts extracted from datetime values")
                    #print(data.head())
                except Exception as e:
                    logging.error(f"Error occurred while extracting date parts: {str(e)}")
                return data
        
        def set_date_column_as_index(self, data, date_column):
            if date_column in data.columns:
                data.set_index(date_column, inplace=True)
                logging.info(f"Index set to the '{date_column}' column.")
                #print(data.head())
            else:
                logging.error(f"Error: '{date_column}' not found in the columns.")
            return data

        def detect_column_types(self, data):
            numerical_columns = []
            categorical_columns = []

            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    numerical_columns.append(column)
                elif data[column].dtype in ['object', 'string']:
                    categorical_columns.append(column)

            logging.info("Numerical columns: %s", numerical_columns)
            logging.info("Categorical columns: %s", categorical_columns)

            #print(data.head())

            return numerical_columns, categorical_columns
               
        # def one_hot_encode_categorical_columns(self, data, categorical_columns):
        #     data[categorical_columns] = data[categorical_columns].astype(str)
        #     encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
        #     data = pd.concat([data, encoded_data], axis=1)
        #     data.drop(columns=categorical_columns, inplace=True)

        #     logging.info(f"One-Hot Encoding applied to categorical columns: {categorical_columns}")
        #     logging.info(f"One-Hot encoded columns appended to the original dataset. Updated dataset shape: {data.shape}")

        #     return data

        def one_hot_encode_categorical_columns(self, data, categorical_columns, drop_first=False):
            data[categorical_columns] = data[categorical_columns].astype(str)
            encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=drop_first)
            
            # Drop duplicated columns
            encoded_data = encoded_data.loc[:, ~encoded_data.columns.duplicated()]
            
            # Optionally drop the original columns
            if not drop_first:
                data.drop(columns=categorical_columns, inplace=True)

            logging.info(f"One-Hot Encoding applied to categorical columns: {categorical_columns}")
            logging.info(f"One-Hot encoded columns appended to the original dataset. Updated dataset shape: {encoded_data.shape}")

            return encoded_data

        def find_target_column(self, data):
            try:
                config = read_process()
                target_column_keywords = config.get('target_column_keywords', [])

                target_columns = []
                for keyword in target_column_keywords:
                    for column in data.columns:
                        if keyword.lower() in column.lower():
                            logging.info(f"Found target column: {column}")
                            target_columns.append(column)

                if not target_columns:
                    logging.warning("Target column not found in the dataset columns.")
                    return None
                
                if len(target_columns) > 1:
                    print("Multiple target columns detected:")
                    for idx, col in enumerate(target_columns):
                        print(f"{idx+1}. {col}")
                    
                    while True:
                        choice = input("Please enter the number corresponding to the target column you want to select: ")
                        try:
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(target_columns):
                                selected_column = target_columns[choice_idx]
                                logging.info(f"User selected target column: {selected_column}")
                                # Update target_columns list with the selected column
                                target_columns = [selected_column]
                                return target_columns
                            else:
                                print("Invalid choice. Please enter a valid number.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                else:
                    return target_columns

            except Exception as e:
                logging.error(f"Error occurred while finding target column: {str(e)}")
                return None

        def remove_special_characters(self, data, target_columns):

            try:
                config = read_process()
                target_column_keywords = config.get('target_column_keywords', ['Target'])

                for target_column in target_columns:
                    data[target_column] = data[target_column].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
                    logging.info(f"Special characters removed from '{target_column}' column.")
                    return data
                
            except Exception as e:
                logging.error(f"Error occurred while removing special characters: {str(e)}")
                return None
            
        def convert_numerical_columns_to_float(self, data, numerical_columns):
            try:
                for column in numerical_columns:
                    data[column] = data[column].astype(float)
                    logging.info(f"Converted '{column}' to float data type.")
                return data
            
            except Exception as e:
                logging.error(f"Error occurred while converting numerical columns to float: {str(e)}")
                return None

        def handle_null_values(self, data):
            try:
                config = read_process()
                missing_values = config.get('missing_values', [])

                for missing_value in missing_values:
                    data.replace(missing_value, pd.NA, inplace=True)

                data.ffill(inplace=True)
                data.bfill(inplace=True)

                null_count = data.isnull().sum().sum()
                logging.info(f"Missing values handled. Null count after treatment: {null_count}")

                return data
            
            except FileNotFoundError as e:
                logging.error(f'Config file not found: {str(e)}')
            except Exception as e:
                logging.error(f'An error occurred: {str(e)}')

        def detect_outliers(self, data, target_columns):
            try:
                outliers = pd.DataFrame(index=data.index, columns=target_columns)
                for target_column in target_columns:
                    z_scores = np.abs((data[target_column] - data[target_column].mean()) / data[target_column].std())
                    outliers[target_column] = z_scores > 3 
                    logging.info(f'Outliers Detected in target column {target_column}')
                return outliers  # Return the outliers DataFrame after processing
            except Exception as e:
                logging.error(f"Error occurred while detecting outliers: {str(e)}")
                return None

        def winsorize_data(self, data, target_columns, lower_percentile=0.05, upper_percentile=0.95):

            winsorized_data = data.copy()

            for target_column in target_columns:
                lower_limit = winsorized_data[target_column].quantile(lower_percentile)
                upper_limit = winsorized_data[target_column].quantile(upper_percentile)
                winsorized_data[target_column] = np.clip(winsorized_data[target_column], lower_limit, upper_limit)

                logging.info(f'Outlier Treatment completed on target column {target_column}')

            return winsorized_data

        
        def correlation_analysis(self, data, target_columns, threshold=0.5):
            correlation_matrix = data.corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
            plt.title('Correlation Matrix')

            config = read_process()  
            save_path = config.get('save_path', [] )

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)

            selected_features = []
            for target_column in target_columns:
                target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
                selected_features.extend(target_correlation[target_correlation > threshold].index.tolist())
            selected_features = list(set(selected_features))
            logging.info(f"Selected features based on correlation with the targets (threshold={threshold}):")
            selected_data = data[selected_features]
            print(selected_data)
            return selected_data

        def normalize_dataset(self, selected_data):
            scaler = MinMaxScaler()
            normalised_data = scaler.fit_transform(selected_data)
            normalised_data = pd.DataFrame(normalised_data, columns= selected_data.columns, index=selected_data.index)
            logging.info("Data Normalization Completed.")
            print(normalised_data)
            return normalised_data
        
        def move_target_to_first(self, normalized_data, target_columns):
            input_columns = list(normalized_data.columns)
            for target_column in target_columns:
                if target_column in input_columns:
                    input_columns.remove(target_column)
                    input_columns.insert(0, target_column)
                    logging.info(f"Moved '{target_column}' to the first position in the input feature list.")
                    print(normalized_data.head())
                else:
                    logging.info("Target column '{}' not found in input columns.".format(target_column))
            return normalized_data

        def convert_to_numpy(self, normalised_data):
            np_data = normalised_data.to_numpy()
            logging.info("DataFrame converted to Numpy Array.")
            print(np_data)
            return np_data
        
        def save_np_data_to_processed_dir(self, np_data):
            config = read_process()
            processed_data_path = config.get('processed_data_path', [])
            os.makedirs(processed_data_path, exist_ok=True)
            np_data_file = os.path.join(processed_data_path, 'preprocessed_data.npy')
            np.save(np_data_file, np_data)
            logging.info("Preprocessed data saved to: %s", np_data_file)