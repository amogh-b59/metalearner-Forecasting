import os
import sys
import logging
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

sys.path.append(project_root)

from src.pipeline.data_transformation_pipeline import DataTransformationPipeline

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Data transformation called")

    preprocessed_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'preprocessed_data.npy'))
    preprocessed_data = np.load(preprocessed_data_path)

    data_pipeline = DataTransformationPipeline(preprocessed_data)

    data_pipeline.build_models()  
    data_pipeline.build_training_data()


if __name__ == "__main__":
    main()