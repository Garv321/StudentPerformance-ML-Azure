
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        train_path, test_path = DataIngestion().initiate_data_ingestion()
        train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
        ModelTrainer().initiate_model_trainer(train_arr, test_arr)




if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()