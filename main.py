import pandas as pd
import pandas_datareader as pdt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout, LSTM 
from tensorflow.python.keras.models import Sequential

TARGET = 'NAT'
SAMPLE_START = dt.datetime(2015, 1, 3)
SAMPLE_END = dt.datetime(2022, 10, 1)


def main():
    data_sample = pdt.DataReader(TARGET, 'yahoo', SAMPLE_START, SAMPLE_END)
    mapper = MinMaxScaler(feature_range=(0,1))
    data_sample_mapped = mapper.fit_transform(data_sample['Close'].values.reshape(-1,1))

    training_data = []
    resulting_data = []

    graphing_data = []
    graphing_result = []

    evaluation_data = []
    evaluation_result_data = []

    prediction_duration = 500

    for x in range(prediction_duration, len(data_sample_mapped)):
        training_data.append(data_sample_mapped[x-prediction_duration: x, 0])
        resulting_data.append(data_sample_mapped[x, 0])

    training_data, resulting_data = np.array(training_data), np.array(resulting_data)
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1], 1))

    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(training_data.shape[1], 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))

    # model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(training_data, resulting_data, epochs=25, batch_size=32)

    # # Breakpoint

    # test_sample_start = dt.datetime(2020, 1, 1)
    # test_sample_end = dt.datetime.now()

    # test_data = pdt.DataReader(TARGET, 'yahoo', test_sample_start, test_sample_end)
    # actual_prices = test_data['Close'].values

    # total_dataset = pd.concat( (data_sample['Close'], test_data['Close']), axis=0 )

    # model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_duration:].values
    # model_inputs = model_inputs.reshape(-1, 1)
    # model_inputs = mapper.transform(model_inputs)

    # print(len(total_dataset), len(test_data))
    # print(len(total_dataset) - len(test_data) - prediction_duration)

    # x_test = []

    # for x in range(prediction_duration, len(model_inputs)):
    #     x_test.append(model_inputs[x-prediction_duration:x, 0])
    
    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predicts = model.predict(x_test)
    # predicts = mapper.inverse_transform(predicts)
    
    # plt.plot(actual_prices, color="orange", label="True")
    # plt.plot(predicts, color="blue", label="Predicted")
    # plt.title(f" {TARGET} test prediction")
    # plt.xlabel('Time')
    # plt.ylabel('INFN share price')

    # plt.legend()
    # plt.show()


    # # Future predictions

    # rd_sample = [model_inputs[len(model_inputs) + 1 - prediction_duration:len(model_inputs+1), 0]]
    # rd_sample = np.array(rd_sample)
    # rd_sample = np.reshape(rd_sample, (rd_sample.shape[0], rd_sample.shape[1], 1))

    # final_val = model.predict(rd_sample)
    # final_val = mapper.inverse_transform(final_val)
    # print(final_val)



if __name__=="__main__":
    main()
