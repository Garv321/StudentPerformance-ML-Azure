
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def initiate_data_ingestion(self):
        df = pd.read_csv("data/student.csv")
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train.to_csv("artifacts/train.csv", index=False)
        test.to_csv("artifacts/test.csv", index=False)
        return "artifacts/train.csv", "artifacts/test.csv"
