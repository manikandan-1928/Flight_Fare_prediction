import os
import pickle
import pandas as pd


class PredictionPipeline:
    def __init__(self,input_data):
        self.input_data =input_data

    
    def predict(self):

        input_data = pd.DataFrame(self.input_data)

        input_data['Date_of_Journey'] = pd.to_datetime(input_data['Date_of_Journey'])
        
        # Extract features from 'Date_of_Journey'
        input_data['Journey_day'] = pd.DatetimeIndex(input_data['Date_of_Journey']).day
        input_data['Journey_month'] = pd.DatetimeIndex(input_data['Date_of_Journey']).month
        input_data['Journey_weekday'] = pd.DatetimeIndex(input_data['Date_of_Journey']).weekday
        input_data['Journey_year'] = pd.DatetimeIndex(input_data['Date_of_Journey']).year
        
        # Modify 'Destination' values
        input_data['Destination'] = np.where(input_data['Destination'] == 'Delhi', 'New Delhi', input_data['Destination'])
        
        # Extract features from 'Dep_Time' and 'Arrival_Time'
        input_data['Dep_Time_Hr'] = input_data['Dep_Time'].str.extract('(\d+):(\d+)').astype(int)[0]
        input_data['Dep_Time_Min'] = input_data['Dep_Time'].str.extract('(\d+):(\d+)').astype(int)[1]
        input_data['Arr_Time_Hr'] = input_data['Arrival_Time'].str.extract('(\d+):(\d+)').astype(int)[0]
        input_data['Arr_Time_Min'] = input_data['Arrival_Time'].str.extract('(\d+):(\d+)').astype(int)[1]
        
        # Extract duration features
        input_data['Duration_Hour'] = input_data['Duration'].str.extract('(\d+)h', expand=False).fillna(0).astype(int)
        input_data['Duration_Minute'] = input_data['Duration'].str.extract('(\d+)m', expand=False).fillna(0).astype(int)
        
        # Map 'Total_Stops' to numerical values
        input_data['Total_Stops'] = input_data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
        
        # Drop unnecessary columns
        input_data.drop(columns=['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Route', 'Additional_Info', 'Duration'], axis=1, inplace=True)

        preprocessor_path = os.path.join('artifacts','data_cleaning','preprocessor.pkl')

        with open(preprocessor_path, 'rb') as file:
            preprocessor_obj = pickle.load(file)

        input_transformed_data = preprocessor_obj.transform(input_data)

        trained_model_path = os.path.join('artifacts','trained_model.h5')
        
        with open(trained_model_path, 'rb') as file:
            model = pickle.load(file)

        price = model.predict([input_transformed_data])

        print(price)

        return price

