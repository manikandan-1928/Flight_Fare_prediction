import os
from src.mlProject import logger
import pandas as pd
import numpy as np
from src.mlProject.entity.config_entity import DataCleaningConfig



class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def data_cleaning(self):
        data = self.load_data()

        if data is not None:
            cleaned_data = self.perform_data_cleaning(data)
            self.save_cleaned_data(cleaned_data)
        else:
            logger.error("Validation failed. Check the input data and schema")

    def load_data(self):
        data_path = self.config.data_path
        try:
            with open(data_path, 'r') as file:
                data = pd.read_csv(file)
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def perform_data_cleaning(self, df):
        logger.info("Data cleaning process started")

        # Make a copy of the DataFrame to avoid modifying the original data
        data = df.copy()

        # Data cleaning operations here...
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
        data['Journey_day'] = pd.DatetimeIndex(data['Date_of_Journey']).day
        data['Journey_month'] = pd.DatetimeIndex(data['Date_of_Journey']).month
        data['Journey_weekday'] = pd.DatetimeIndex(data['Date_of_Journey']).weekday
        data['Journey_year'] = pd.DatetimeIndex(data['Date_of_Journey']).year
        data['Destination'] = np.where(data['Destination'] == 'Delhi', 'New Delhi', data['Destination'])
        data['Dep_Time_Hr'] = data['Dep_Time'].str.extract('(\d+):(\d+)').astype(int)[0]
        data['Dep_Time_Min'] = data['Dep_Time'].str.extract('(\d+):(\d+)').astype(int)[1]
        data['Arr_Time_Hr'] = data['Arrival_Time'].str.extract('(\d+):(\d+)').astype(int)[0]
        data['Arr_Time_Min'] = data['Arrival_Time'].str.extract('(\d+):(\d+)').astype(int)[1]
        data['Duration_Hour'] = data['Duration'].str.extract('(\d+)h', expand=False).fillna(0).astype(int)
        data['Duration_Minute'] = data['Duration'].str.extract('(\d+)m', expand=False).fillna(0).astype(int)
        data['Total_Stops'] = data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
        data.drop(columns=['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Route', 'Additional_Info', 'Duration'], axis=1,
                  inplace=True)

        logger.info("Data cleaning process completed")
        return data

    def save_cleaned_data(self, df):
        try:
            cleaned_data_path = os.path.join(self.config.root_dir, "cleaned_data.csv")
            df.to_csv(cleaned_data_path, index=False)
            logger.info("Cleaned data is saved in artifacts folder.")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
