import os
import sys
import logging
import yaml

# Add the parent directory of 'src' to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

# Now import the necessary modules
from src.utils import read_process
from src.pipeline.data_quality_pipeline import DataQualityCheck

def main():

    data_quality_pipeline = DataQualityCheck(None)
   
    data = data_quality_pipeline.read_data(file_type='excel')
    if data is None:
        logging.error("Failed to read data.")
        return
    
    completeness_scores = data_quality_pipeline.check_completeness(data)
    if completeness_scores is not None:
        logging.info("Completeness Scores: %s", completeness_scores)
    else:
        logging.error("Failed to check completeness.")
  
    duplicate_percentage = data_quality_pipeline.check_duplicates(data)
    if duplicate_percentage is not None:
        logging.info("Duplicate Percentage: %s", duplicate_percentage)
    else:
        logging.error("Failed to check duplicates.")

    uniqueness_percentage = data_quality_pipeline.check_uniqueness(data)    
    if uniqueness_percentage is not None:
        logging.info("Uniqueness Score: %.2f%%", uniqueness_percentage)
    else:
        logging.error("Failed to check unique values.")

    size_percentage = data_quality_pipeline.check_dataset_size(data)
    if size_percentage is not None:
        logging.info("Dataset size: %.2f%%", size_percentage)
    else:
        logging.error("Failed to check data size. ")

    overall_data_quality_score = data_quality_pipeline.calculate_data_quality_score(completeness_scores, duplicate_percentage, uniqueness_percentage, size_percentage)
    if overall_data_quality_score is not None:
        logging.info("Overall Data Quality Score: %.2f%%", overall_data_quality_score)
    else:
        logging.error("Failed to check overall data quality score.")
        
if __name__ == "__main__":
    main()