from pyexpat import model
from matplotlib import test
import pandas as pd
import pandas_datareader as pdt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout, LSTM 
from tensorflow.python.keras.models import Sequential

TARGET = 'INFN'
SAMPLE_START = dt.datetime(2019, 1, 1)
SAMPLE_END = dt.datetime(2022, 9, 1)


def main():
    data_sample = pdt.DataReader(TARGET, 'yahoo', SAMPLE_START, SAMPLE_END)

    print(data_sample)

    mapper = MinMaxScaler(feature_range=(0,1))
    data_sample_mapped = mapper.fit_transform(data_sample['Close'].values.reshape(-1,1))

    prediction_duration = 60

    x_train = []
    y_train = []

    for x in range(prediction_duration, len(data_sample_mapped)):
        x_train.append(data_sample_mapped[x-prediction_duration: x, 0])
        y_train.append(data_sample_mapped[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    print("Thing", x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    x_test = x_train
    actual_prices = data_sample #x_train['Close'].values
    
    predicts = model.predict(x_test)
    predicts = mapper.inverse_transform(predicts)
    
    plt.plot(actual_prices, color="orange", label="True")
    plt.plot(predicts, color="blue", label="Predicted")
    plt.title("INFN test prediction")
    plt.xlabel('Time')
    plt.ylabel('INFN share price')

    plt.legend()
    plt.show()


    
    

    


if __name__=="__main__":
    main()
