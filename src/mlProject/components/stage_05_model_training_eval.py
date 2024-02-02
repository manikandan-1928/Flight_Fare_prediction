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
from sklearn.linear_model import Ridge
import mlflow
from urllib.parse import urlparse
from src.mlProject.entity.config_entity import DataTuningConfig




class ModelTuningEvaluate:
    def __init__(self, config: DataTuningConfig):
        self.config = config

    def grid_search_cv(self):
        try:

            train_data = pd.read_csv(self.config.train_data_path)

            test_data = pd.read_csv(self.config.test_data_path)

            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            best_model = XGBRegressor()        

            params = {
                    #'learning_rate': [0.001, 0.01, 0.1],
                    #'n_estimators': [100, 200, 300, 400]
                    #'max_depth': [3, 6, 9, 12],
                    #'min_child_weight': [1, 3, 5],
                    #'subsample': [0.8, 0.9, 1.0],
                    #'colsample_bytree': [0.8, 0.9, 1.0],
                    #'gamma': [0, 0.1, 0.2],
                    #'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0],
                }


            gs = GridSearchCV(best_model, params, cv=5)
            gs.fit(X_train, y_train)
            
            best_params = gs.best_params_
            print(best_params)
            # Save the best parameters to a JSON file

            param_file_path = os.path.join(self.config.root_dir, "params_report.json")

            # Check if the file already exists
            if os.path.exists(param_file_path):
                # Read the existing JSON data
                with open(param_file_path, 'r') as file:
                    existing_data = json.load(file)
            else:
                # If the file doesn't exist, initialize existing_data as an empty dictionary
                existing_data = {}

            # Update the existing_data with the new data
            existing_data.update(best_params)

            # Write the updated data back to the JSON file
            with open(param_file_path, 'w') as file:
                json.dump(existing_data, file)
             
            return X_train, y_train, X_test, y_test, best_params

        except Exception as e:
            raise e
        


class Training:
    def __init__(self, config):
        self.config = config

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            model_filepath = os.path.join(self.config.root_dir, 'trained_model.pkl')

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

    def log_into_mlflow(self, results, best_model):

        try:

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)

                # Log metrics
                mlflow.log_metrics(results)

                # Log the model into MLflow
                if tracking_url_type_store != "file":
                    # Register the model in the Model Registry
                    mlflow.xgboost.log_model(best_model, "model", registered_model_name="XGBRegressor")
                else:
                    mlflow.xgboost.log_model(best_model, "model")

        except Exception as e:
            raise e

                
    