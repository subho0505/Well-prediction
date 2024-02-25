# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to load and preprocess data
@st.cache  # This decorator allows Streamlit to cache the data loading and preprocessing steps
def load_data():
    # Assuming the data is loaded from a local path
    file_paths = [
        '2017.xlsx',
        '2018-2019.xlsx',
        '2020-2021.xlsx',
        '2022.xlsx',
    ]
    list_data = []
    for file_path in file_paths:
        datas = pd.read_excel(file_path, header=5)
        list_data.append(datas)
    data = pd.concat(list_data, ignore_index=True)
    data_drop = data.dropna()
    data_drop['Date'] = pd.to_datetime(data_drop['Date'])
    data_drop['Month'] = data_drop['Date'].dt.month
    return data_drop

# Function to fit ARIMA model and make predictions
def predict_groundwater_levels(district, data):
    monthly_means = data.groupby(['District', 'Month'])['GW Level(mbgl)'].mean().reset_index()
    district_data = monthly_means[monthly_means['District'] == district]
    model = sm.tsa.ARIMA(district_data['GW Level(mbgl)'], order=(5, 1, 0))
    model_fit = model.fit()
    n_steps = 4  # Predict the next 4 months
    future_predictions_series = model_fit.forecast(steps=n_steps)  # Updated line
    future_predictions_array = future_predictions_series.values  # Convert Series to numpy array
    return future_predictions_array

# Streamlit UI
st.title("Groundwater Level Prediction")

# Load data
data = load_data()

# User input for district name
district = st.selectbox("Select District", data['District'].unique())

# Get predictions
predictions = predict_groundwater_levels(district, data)

# Plot predictions
fig, ax = plt.subplots()
months = range(1, 5)  # Assume the predictions are for the next 4 months
ax.plot(months, predictions, label='Predictions')
ax.set_xlabel('Month')
ax.set_ylabel('GW Level (mbgl)')
ax.set_title(f'Monthly Predictions for GW Levels in {district} (Next 4 Months)')
ax.legend()
ax.grid(True)
st.pyplot(fig)
