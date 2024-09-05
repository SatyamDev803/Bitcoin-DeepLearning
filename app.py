import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

start = datetime(2014, 9, 17)
end = datetime.now().date().isoformat()

st.title('Cryptocurrency Price Prediction')

user_input = st.text_input('Enter Crypto Ticker', 'BTC-USD')

data = yf.download(user_input, start = start, end = end)

# Describing Data
st.subheader(f'Data from {start} - {end}')
st.write(data.describe())

# Visulizations
st.subheader('Closing Price vs Time Chart')
fig1 = px.line(data.Close, color_discrete_map={'Close':'blue'})
st.plotly_chart(fig1)
data['ma100'] = data['Close'].rolling(window=100).mean()
data['ma200'] = data['Close'].rolling(window=200).mean()
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig4 = px.line(data, x=data.index, y=['Close', 'ma100', 'ma200'], color_discrete_map={'Close': 'blue', 'ma100':'yellow', 'ma200': 'red'})
st.plotly_chart(fig4)

prediction_days = 80
future_day = 15

scaler = MinMaxScaler(feature_range=(0,1))

# Load Model
model = load_model('my_model.keras')

#Testing part
start = datetime(2020, 1, 1)
end = datetime.now().date().isoformat()
test_data = yf.download(user_input, start=start, end=end)
st.subheader('Test Data for Prediction')
fig_test_data = px.line(test_data.Close, color_discrete_map={'Close':'blue'})
st.plotly_chart(fig_test_data)

actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

st.subheader(f'{user_input} Predicted Price Vs Actual Price Using LSTM Model (Future {future_day} days)')
# Create a Plotly figure
fig_lstm = go.Figure()
# Add trace for actual prices
fig_lstm.add_trace(go.Scatter(x=test_data.index, y=actual_prices.flatten(), mode='lines', name='Actual Prices', line=dict(color='blue')))
# Add trace for predicted prices
fig_lstm.add_trace(go.Scatter(x=test_data.index, y=prediction_prices.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))
fig_lstm.update_layout(xaxis_title='Date',
                  yaxis_title='Price',
                  legend=dict(x=0, y=1.1, orientation='h'),
                  margin=dict(l=0, r=0, t=50, b=0))
# Display the plot
st.plotly_chart(fig_lstm)

# Predicting Next Day Price 
real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs)+1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

st.subheader('Next Day Price Prediction')
st.write(end)
st.write(prediction)


