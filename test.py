import pandas as pd
import numpy as np
import tensorflow as tf

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/'
'master/iris.csv') 

# #print(iris)

# # one line split 
# train, validation, test = np.split(iris.sample(frac=1), [int(.6*len(iris)),
# int(.8*len(iris))])


# print( [train.columns[i] for i in range(train.shape[1]-1) ] )

# """
# print([int(.6*len(iris)), int(.8*len(iris))])
# print(train)
# print(test)
# print(validation)
# """

# # Assign the train split
# X_train = train[[train.columns[i] for i in range(train.shape[1]-1) ]]
# y_train = train[train.columns[-1]]
# # Assign the test split
# X_test = test[[test.columns[i] for i in range(train.shape[1]-1) ]]
# y_test = test[test.columns[-1]]
# # Assign the validation split
# X_val = validation[[validation.columns[i] for i in range(validation.shape[1]-1) ]]
# y_val = validation[validation.columns[-1]]

import datetime as dt
import pandas_datareader as pdt
import pandas as pd

TARGET = "NAT"

test_sample_start = dt.datetime(2020, 1, 1)
test_sample_end = dt.datetime.now()

test_data = pdt.DataReader(TARGET, 'yahoo', test_sample_start, test_sample_end)
actual_prices = test_data['Close'].values

SAMPLE_START = dt.datetime(2015, 1, 3)
SAMPLE_END = dt.datetime(2022, 10, 1)

data_sample = pdt.DataReader(TARGET, 'yahoo', SAMPLE_START, SAMPLE_END)

total_dataset = pd.concat( (data_sample['Close'], test_data['Close']), axis=0 )

model_inputs = total_dataset[len(total_dataset) - len(test_data) - 500:]

print(type(total_dataset))
print(len(total_dataset) - len(test_data) - 500)
print(len(total_dataset[len(total_dataset) - len(test_data) - 500:]))
print(model_inputs)