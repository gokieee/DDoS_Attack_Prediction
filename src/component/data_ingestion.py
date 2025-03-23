import sys, os
import pandas as pd
import numpy as np 

from src.Exeption import CustomException
from src.logging import logging
from src.config.config import DataIngestionConfig, DataIngestionArtifact
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_raw_data(self) -> pd.DataFrame:
        try:
            logging.info("Loading Dataframe...")
            dataframe = pd.read_csv(r'C:\Users\GOKUL\Desktop\DDoS_Attack_Prediction\Network_Traffic_data\DDos_final.csv')
            
            # Remove duplicates
            logging.info(f"Duplicate rows before removal: {dataframe.duplicated().sum()}")
            dataframe.drop_duplicates(inplace=True)
            logging.info(f"Duplicate rows after removal: {dataframe.duplicated().sum()}")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data), exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.raw_data, index=False)

            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_train_and_test(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Splitting train and test data...")
            train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.data_ingestion_config.train_data_path, index=False)
            logging.info("Train data saved.")

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logging.info("Test data saved.")

            return train, test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_raw_data()
            self.split_train_and_test(dataframe)
            artifact=DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
                raw_data=self.data_ingestion_config.raw_data
            )
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
