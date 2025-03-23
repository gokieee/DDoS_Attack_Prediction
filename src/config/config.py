from dataclasses import dataclass
import sys, os



@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str
    raw_data: str

@dataclass
class DataIngestionConfig:
    raw_data : str = os.path.join("artifact", "ddos_raw.csv")
    train_data_path : str = os.path.join("artifact", "ddos_train.csv")
    test_data_path : str = os.path.join("artifact", "ddos_test.csv")


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str    

Train_File_Name="train.csv"
Test_File_Name="test.csv"
Preprocessor="Preprocessor.pkl"
@dataclass
class DataTransformationConfig:
    preprocessor : str = os.path.join("artifact", Preprocessor)
    transformed_train_file_path: str = os.path.join("artifact", Train_File_Name.replace("csv", "npy"),)
    transformed_test_file_path: str = os.path.join("artifact", Test_File_Name.replace("csv", "npy"), )



@dataclass
class ModelTrainerConfig:
    best_model : str = os.path.join("artifact", "best_model.pkl")