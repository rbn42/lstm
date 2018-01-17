from data_model import load_data
from val import AIAO
import numpy as np
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import losses
from keras import backend as K

seq_len = 30  # 50
batch_size = 64  # 512
max_epoch = 100
lstm_units = 128

print('> Loading data... ')
X_train, y_train, X_test, y_test = load_data('./data/GSPC.csv', seq_len, True)
print('> Data Loaded. Compiling...')

model = Sequential()
model.add(LSTM(
    input_shape=(seq_len, 6),
    output_dim=lstm_units,
    return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))

# 暂且用sigmoid限制0-1范围
# model.add(Activation("sigmoid"))

# loss设定估计是个问题,下面都是直接加入y,然后对比差值的,但是我们需要不一样的算法
model.compile(loss='mse', optimizer="rmsprop")
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    nb_epoch=max_epoch,
    validation_split=0.05)

AIAO(model, X_test, y_test, seq_len, )
