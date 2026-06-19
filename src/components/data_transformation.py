import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        
        # Load data
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # Define target column (IMPORTANT)
        TARGET_COLUMN = "math_score"

        # Split features and target
        X_train = train.drop(TARGET_COLUMN, axis=1)
        y_train = train[TARGET_COLUMN]

        X_test = test.drop(TARGET_COLUMN, axis=1)
        y_test = test[TARGET_COLUMN]

        #  Handle categorical data (VERY IMPORTANT)
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        # Align columns (to avoid mismatch)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        #  Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save preprocessor
        pickle.dump(scaler, open("artifacts/preprocessor.pkl", "wb"))

        # Combine features + target
        train_arr = np.c_[X_train_scaled, y_train]
        test_arr = np.c_[X_test_scaled, y_test]

        return train_arr, test_arr, "artifacts/preprocessor.pkl"