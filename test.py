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

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()