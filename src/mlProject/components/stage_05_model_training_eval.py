import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from src.mlProject import logger
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import mlflow
from urllib.parse import urlparse
from src.mlProject.entity.config_entity import DataTuningConfig



class Training:
    def __init__(self, config: DataTuningConfig):
        self.config = config

    def train_model(self):
        try:

            train_data = pd.read_csv(self.config.train_data_path)

            test_data = pd.read_csv(self.config.test_data_path)

            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            model_filepath = os.path.join(self.config.root_dir, 'trained_model.h5')

            # Load the JSON file
            with open(self.config.model_report, 'r') as file:
                data = json.load(file)

            # Extract the best_model_name
            best_model_name = data["best_model_name"]

            print(best_model_name)

            if best_model_name == 'Random Forest':
                best_model = RandomForestRegressor()

            elif best_model_name == 'Linear Regression':
                best_model = LinearRegression()

            elif best_model_name == 'XGBRegressor':
                best_model = XGBRegressor()

            best_model.set_params(**self.config.all_params)

            best_model.fit(X_train, y_train)
            print('Model Trained')

            # Save the trained model
            with open(model_filepath, 'wb') as file:
                pickle.dump(best_model, file)

            # Load existing results if available
            results_path = os.path.join(self.config.root_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as file:
                    existing_results = json.load(file)
            else:
                existing_results = {}

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_r2_score = r2_score(y_train, y_train_pred)

            test_model_mae = mean_absolute_error(y_test, y_test_pred)
            test_model_mse = mean_squared_error(y_test, y_test_pred)
            test_model_r2_score = r2_score(y_test, y_test_pred)

            new_results = {
                "train_data_r2_score": train_model_r2_score,
                "test_data_r2_score": test_model_r2_score,
                "test_data_mae": test_model_mae,
                "test_data_mse": test_model_mse,
            }

            # Update existing results with new results
            existing_results.update(new_results)

            # Write updated results back to the file
            with open(results_path, 'w') as file:
                json.dump(existing_results, file)

            return new_results, best_model
        
        except Exception as e:
            raise e
