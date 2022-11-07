import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date
import os

pd.options.display.float_format = '{:.10f}'.format

df = web.DataReader('AAPL', data_source='yahoo', start='1980-12-12', end=date.today())

plt.figure(figsize=(16,8))
plt.title('AAPL Price History')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.plot(df['Close'])
# plt.show()

data = df.filter(['Close'])
dataset =  data.to_numpy()
training_len = math.ceil(len(dataset) * .8)


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


train_data = scaled_data[0:training_len, :]
x_train = []
y_train =[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


