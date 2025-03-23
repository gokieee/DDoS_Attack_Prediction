import sys, os

from src.config.config import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataIngestionArtifact
)
from src.Exeption import CustomException
from src.logging import logging
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation




class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            
            data_ingestion = DataIngestion(DataIngestionConfig())  # Ensure proper config is passed
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifact):

        try:
            logging.info("Starting data transformation")

            data_transformation = DataTransformation(
                data_transformation_config=DataTransformationConfig(),
                 data_ingestion_artifact=data_ingestion_artifact  # Pass the artifact correctly
                )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Data Transformation completed successfully.")
            return data_transformation_artifact

        except Exception as e:
           raise CustomException(e, sys)   
        
    def run_pipeline(self):
        try:
            logging.info("Running training pipeline...")
            
            data_ingestion_artifact = self.start_data_ingestion()
            print(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            print(data_transformation_artifact)
            
            logging.info("Training pipeline completed successfully.")
            return (
                data_ingestion_artifact,
                data_transformation_artifact
            )
        
        except Exception as e:
            raise CustomException(e, sys)
  