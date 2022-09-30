from pyexpat import model
import pandas as pd
import pandas_datareader as pdt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Deopout, LSTM 
from tensorflow.python.keras.models import Sequential

TARGET = 'INFN'
SAMPLE_START = dt.datetime(2019, 1, 1)
SAMPLE_END = dt.datetime(2022, 9, 1)


def main():
    data_sample = pdt.DataReader(TARGET, 'yahoo', SAMPLE_START, SAMPLE_END)

    mapper = MinMaxScaler(feature_range=(0,1))
    data_sample_mapped = mapper.fit_transform(data_sample['Close'].values.reshape(-1,1))

    prediction_duration = 1097

    x_train = []
    y_train = []

    for x in range(prediction_duration, len(data_sample_mapped)):
        x_train.append(data_sample_mapped[x-prediction_duration: x, 0])
        y_train.append(data_sample_mapped[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add()


if __name__=="__main__":
    main()
