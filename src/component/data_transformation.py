import sys, os
import numpy as np

import pandas as pd 

from imblearn.combine import SMOTEENN
from collections import Counter

from sklearn.preprocessing import (
    LabelEncoder,
    FunctionTransformer,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.Exeption import CustomException
from src.logging import logging
from src.config.config import (
    DataTransformationConfig,
    DataTransformationArtifact, 
    DataIngestionArtifact
)
from src.utils.utils import (
    save_numpy_array_data,
    save_object
)

class DataTransformation:
    def __init__(self, data_transformation_config : DataTransformationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact):

        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact= data_ingestion_artifact
        except Exception as e:
            raise CustomException(sys, e)
        
    @staticmethod   
    def read_csv(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(sys, e)    
        
    def get_data_transformer_object(self)->Pipeline:

        try:
            numerical_features=[' Packet Length Std', ' Bwd Packet Length Mean',
                               'Bwd Packet Length Max', ' Average Packet Size',
                               ' Packet Length Variance', ' Packet Length Mean', ' Max Packet Length',
                               ' Bwd Packet Length Std', ' Avg Bwd Segment Size'
                               ]
            scale:StandardScaler=StandardScaler()
            functiontransformer:FunctionTransformer=FunctionTransformer(np.cbrt, validate=True)
            scale_pipeline:Pipeline=Pipeline([("scale", scale)])
            functiontransformer_pipeline:Pipeline=Pipeline([("functiontransform", functiontransformer)])
            preprocessor:ColumnTransformer=ColumnTransformer([
                ("scale_pipeline", scale_pipeline, numerical_features),
                ("funtiontransfomer", functiontransformer_pipeline, numerical_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(sys, e)

    def remove_outliers_iqr(self, col, df):

        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit),col] = upper_limit
            df.loc[(df[col]<lower_limit),col] = lower_limit
            return df
        except Exception as e:
            raise CustomException(sys, e)

    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            logging.info("Entered initiate_data_transformation method of DataTransformation class")

            train_df = DataTransformation.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_csv(self.data_ingestion_artifact.test_file_path)

            numerical_features = [
            ' Packet Length Std', ' Bwd Packet Length Mean', 'Bwd Packet Length Max',
            ' Average Packet Size', ' Packet Length Variance', ' Packet Length Mean',
            ' Max Packet Length', ' Bwd Packet Length Std', ' Avg Bwd Segment Size'
            ]
            TARGET = [' Label']

            # Remove outliers
            for feature in numerical_features:
                self.remove_outliers_iqr(feature, train_df)
                self.remove_outliers_iqr(feature, test_df)

            logging.info("Outlier removal completed")

            # Prepare input and target features
            input_feature_train_df = train_df.drop(columns=TARGET[0], axis=1)
            target_feature_train_df = train_df[TARGET[0]]

            input_feature_test_df = test_df.drop(columns=TARGET[0], axis=1)
            target_feature_test_df = test_df[TARGET[0]]

            labelencoder=LabelEncoder()
            target_feature_train_encoded=labelencoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = labelencoder.transform(target_feature_test_df)
            save_object("final_object/label_encoder.pkl", labelencoder)

            # Get preprocessor object
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            # Apply transformations
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN on Training dataset")

            smt = SMOTEENN(sampling_strategy="minority")

            input_features_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_encoded
            )
            input_features_test_final, target_feature_test_final = smt.fit_resample(
               transformed_input_test_feature, target_feature_test_encoded
            )
            logging.info(f"Label distribution in training data after SMOTEENN:\n{np.bincount(target_feature_train_final)}")
            logging.info(f"Label distribution in test data after SMOTEENN:\n{np.bincount(target_feature_test_final)}")

            logging.info("SMOTEENN applied successfully")

            # Save transformed arrays
            train_arr = np.c_[
                input_features_train_final, 
                np.array(target_feature_train_final)
            ]
            test_arr = np.c_[
                input_features_test_final, 
                np.array(target_feature_test_final)
            ]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save preprocessor object
            save_object(self.data_transformation_config.preprocessor, preprocessor_object)
            save_object("final_object/preprocessor.pkl", preprocessor_object)

            # Create transformation artifact
            transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.preprocessor,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return transformation_artifact
        except Exception as e:
          raise CustomException(e, sys)
