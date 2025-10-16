import pandas as pd
import numpy as np
import sklearn
# from IPython.display import display


from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import MeanEncoder

import joblib
import datetime
import xgboost

from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.ensemble import RandomForestRegressor


# Display settings
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output = 'pandas') # To display sklearn outputs as pandas DataFrames

# Reading any one dataset
path = r"Datasets/train_data.csv"
df = pd.read_csv(path)

# For Web Application
import streamlit as st

st.set_page_config(page_title = "Flight Price Prediction",
                    layout = "wide",
                    page_icon = "✈️")

# -----------------------------------------------------------------------------------------------------------------------------------
import sys
st.write(sys.executable)



# -----------------------------------------------------------------------------------------------------------------------------------

st.title("Flight Price Prediction App ✈️ - With AWS Sagemaker")

# -----------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    # Airline
    airline_options = df['airline'].unique().tolist()
    airline = st.selectbox(label = 'Airline', options = airline_options, index = None, placeholder = 'Select Airline')

with col2:
    # Date of journey
    todays_date = datetime.date.today()
    max_date = datetime.timedelta(days = 90) + todays_date
    doj = st.date_input(label = 'Date of Journey', value = None, min_value = todays_date)

col3, col4 = st.columns(2)

with col3:
    # Source
    source_options = df['source'].unique().tolist()
    source = st.selectbox(label = 'Source', options = source_options, index = None, placeholder = 'Select Source City')

with col4:
    # Destination
    dest_options = df['destination'].unique().tolist()
    destination = st.selectbox(label = 'Destination', options = dest_options, index = None, placeholder = 'Select Destination City')

col5, col6, col7 = st.columns(3)

with col5:
    # Departure and Arrival time
    dep_time = st.time_input(label = 'Departure Time', value = None, help = 'Select Departure Time')
    arrival_time = st.time_input(label = 'Arrival Time', value = None, help = 'Select Arrival Time')

with col6:
    # Duration
    duration = st.number_input(label = 'Duration (in minutes)',
                            min_value = 0,
                            value = None,
                            step = 1,
                            placeholder = 'Enter Duration in Minutes')
with col7:
    # Total Stops
    total_stops = st.number_input(
        label = 'Total Stops',
        min_value = 0,
        value = None,
        step = 1,
        placeholder = 'Enter Total Stops')

    
# -----------------------------------------------------------------------------------------------------------------------------------

data = (
    {
        'airline': [airline],
        'date_of_journey': [doj_str],
        'source': [source],
        'destination': [destination],
        'dep_time': [dep_time_str],
        'arrival_time': [arrival_time_str],
        'duration': [duration],
        'total_stops': [total_stops]
    }
)

# type = {'date_of_journey': str, 'dep_time': str, 'arrival_time': str}
user_input = pd.DataFrame(data)   # .astype(type)
convert = {
    'airline': str,
    'date_of_journey': str,
    'source': str,
    'destination': str,
    'dep_time': str,
    'arrival_time': str
}
st.dataframe(user_input.astype(convert)

st.write(user_input.dtypes)

# -----------------------------------------------------------------------------------------------------------------------------------

if st.button('Predict Price'):

    try:

        # loading preprocessor and preprocessing the user_input data
        import cloudpickle
        with open("flights_preprocessor.pkl", 'rb') as f:
            preprocessor = cloudpickle.load(f)

        preprocessed_data = preprocessor.transform(user_input)

        # loading model and predicting the price
        with open("xgboost-flight-price-model", 'rb') as f:
            model = joblib.load(f)
        
        # Converting to Dmatrix
        dmatrix = xgboost.DMatrix(preprocessed_data)
        prediction = model.predict(dmatrix)

        st.success(f"Predicted Flight Price is: {round(prediction[0])} INR")

    except Exception as e:

        st.error(f"Please enter all the valid details")






