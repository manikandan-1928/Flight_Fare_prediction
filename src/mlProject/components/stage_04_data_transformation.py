import os
import pandas as pd
import numpy as np
import pickle
from src.mlProject import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation(self):

        try:
            numerical_columns = ["Total_Stops","Journey_day","Journey_month","Journey_weekday","Journey_year","Dep_Time_Hr",
                                 "Dep_Time_Min","Arr_Time_Hr","Arr_Time_Min","Duration_Hour","Duration_Minute"]
            categorical_columns = [
                "Airline",
                "Source",
                "Destination",
                
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )

            with open('preprocessor.pkl', 'wb') as file:
                pickle.dump(preprocessor, file)

            return preprocessor
        
        except Exception as e:
            raise e 
       
    def initiate_data_transformation(self):
        try:
            df = pd.read_csv("artifacts/data_cleaning/cleaned_data.csv")

            print(df.shape)

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation()
            target_column_name = "Price"

            numerical_columns = ["Total_Stops", "Journey_day", "Journey_month", "Journey_weekday", "Journey_year",
                                 "Dep_Time_Hr", "Dep_Time_Min", "Arr_Time_Hr", "Arr_Time_Min", "Duration_Hour",
                                 "Duration_Minute"]

            categorical_columns = [
                "Airline",
                "Source",
                "Destination",
            ]

            input_feature_df = df.drop(columns=[target_column_name], axis=1)
            target_feature_df = df[target_column_name]

            print(input_feature_df.shape)
            

            logger.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            logger.info("Saved preprocessing object.")

            input_arr = np.c_[
                input_feature_arr, np.array(target_feature_df)
            ]

            transformed_data_df = pd.DataFrame(input_arr)
            
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save the transformed data to CSV
            transformed_data_path = os.path.join(self.config.root_dir, "transformed_data.csv")
            transformed_data_df.to_csv(transformed_data_path, index=False)
            logger.info(f"Transformed data saved at: {transformed_data_path}")


        except Exception as e:
            raise e



