# importing all the important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


company = 'AAPL'  # Share Name of Apple Stock
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2022, 1, 1)

data = web.DataReader(company, 'yahoo', start, end) # Loading data from the yahoo finance public data

# preparing data for the neural network

scaler = MinMaxScaler(feature_range=(0, 1))       # scaling the prices in range from 0 to 1 
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) # taking the closing price of the day and reshaping it in the network  

##IMPORTANT 
# we aim to get the graph of the moving average which is a statistic that captures the average change in a data series over time, 
# so we need to give the model a specific period of time to take the average so here I take it as 60 days


prediction_days = 60 

x_train = []   # preparing the training data here
y_train = [] 

for x in range(prediction_days, len(scaled_data)):         # taking range from 60 up until the last index
    x_train.append(scaled_data[x-prediction_days:x, 0])     # feeding the value of x-axis which is the time to the training model with each iteration
    y_train.append(scaled_data[x, 0])                       # feeding the value of y-axis which is the price to the trainig model

x_train, y_train = np.array(x_train), np.array(y_train)    # coverting them into numpy arrays 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # reshaping it according to the neural network which is 0 and 1

# building model using LSTM (Long short term memory) layers
model = Sequential() 

model.add(LSTM(units=50, return_sequences=True,
          input_shape=(x_train.shape[1], 1))) #adding the LSTM layer with the price value 
model.add(Dropout(0.2)) #dropping the layer
model.add(LSTM(units=50, return_sequences=True)) 
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  #Densing the final layer, which is the prediction price of the stock

model.compile(optimizer='adam', loss='mean_squared_error') #Compiling the data using the adam optimizer and giving loss asa the mean squared error
model.fit(x_train, y_train, epochs=25, batch_size=32) # training the model 25 times with the same values taken 32 at a time

# testing model accuracy on existing data

test_start = dt.datetime(2022, 1, 1)
test_end = dt.datetime.now()
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0) #concatinating the data 

model_inputs = total_dataset[len(
    total_dataset) - len(test_data) - prediction_days:].values 
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs) #scaling the data again with respect to the test data time

# make predictions on test data 

x_test = []

for x in range(prediction_days, len(model_inputs)):  
    x_test.append(model_inputs[x-prediction_days:x, 0]) 

# taking the predicted data and feeding it to the numpy array 

x_test = np.array(x_test) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# now feeding the data to the model we created 
predicted_prices = model.predict(x_test) 
predicted_prices = scaler.inverse_transform(predicted_prices) #inverse transforming the x-test array

# plot the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green',
         label=f"""Predicted Moving Average of {company} Price
                    (If this line is pointing up, the price will go up 
                    If this line is pointing down, the price will go down)""")  # if the line is going up, the price of the share will go up as well and if the line is going down, the price will go down
plt.title(f"{company} Share Price")
plt.xlabel('Time (days)')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()
##


