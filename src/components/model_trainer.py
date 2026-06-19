import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student_Performance_Project")

class ModelTrainer:
    def initiate_model_trainer(self, train_array, test_array):

        X_train, y_train = train_array[:, :-1], train_array[:, -1]
        X_test, y_test = test_array[:, :-1], test_array[:, -1]

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5]
        }

        grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)

        with mlflow.start_run(run_name="student_model_run"):

            print(" MLflow Run Started")

            grid.fit(X_train, y_train)
            model = grid.best_estimator_

            preds = model.predict(X_test)

            #  REGRESSION METRICS
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            print("R2 Score:", r2)
            print("RMSE:", rmse)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(model, "model")

            print("MLflow Logging Done")

        pickle.dump(model, open("artifacts/model.pkl", "wb"))