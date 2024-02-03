from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import pandas as pd
from src.mlProject.pipeline.predict import PredictionPipeline
from datetime import datetime
import numpy as np


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self, input_data=None):
        # Initialize PredictionPipeline with input_data
        self.predict = PredictionPipeline(input_data)

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    Airline = request.form.get('airline')
    Source = request.form.get('Source')
    Destination = request.form.get('Destination')
    Total_Stops = request.form.get('stops')
    Date_of_Departure = request.form.get('Dep_date')
    Date_of_Arrival = request.form.get('Arrival_date')
    Dep_Time = request.form.get('Dep_Time')
    Arrival_Time = request.form.get('Arrival_Time')

    # Combine date and time strings and convert to datetime objects
    departure_datetime = datetime.strptime(f'{Date_of_Departure} {Dep_Time}', '%Y-%m-%d %H:%M')
    arrival_datetime = datetime.strptime(f'{Date_of_Arrival} {Arrival_Time}', '%Y-%m-%d %H:%M')

   # Calculate duration in hours and minutes
    duration_minutes = (arrival_datetime - departure_datetime).total_seconds() // 60

    # Calculate hours and minutes
    hours, minutes = divmod(duration_minutes, 60)

    # Format duration as 'hours' and 'minutes'
    Duration = f'{int(hours)}h {int(minutes)}m'

    print(Duration)

    # Creating a dictionary with input values
    input_dict = {
        'Airline': [Airline],
        'Source': [Source],
        'Destination': [Destination],
        'Total_Stops': [Total_Stops],
        'Date_of_Departure': [Date_of_Departure],
        'Dep_Time': [Dep_Time],
        'Arrival_Time': [Arrival_Time],
        'Duration': [Duration]
    }



    # Creating a DataFrame from the dictionary
    input_df = pd.DataFrame(input_dict)

    # Initialize ClientApp and pass input_df to PredictionPipeline
    clApp = ClientApp(input_data=input_df)

    # Perform prediction
    results = clApp.predict.predict()
    return render_template('index.html', results= str(np.round(results[0],2)))
                           


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #for AWS