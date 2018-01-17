import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def load_data(filename, seq_len, normalise_window):
    data = [[float(i) for i in line.split(',')[1:]] for line in open(filename)]
    data = np.asarray(data)
    # print(data.shape)
    # data0=data[:-1]
    # close0=data0[:,3,np.newaxis] #收盘价
    # vol0=data0[:,5,np.newaxis] #成交量
    # data=data[1:]
    #data[:,:5] /= close0
    #data[:,5:] /= vol0
    # 如上的复杂处理去除掉,因为应该可以信赖网络自己调整到正确的基数

    # 比例换算
    data = data[1:] / data[:-1]
    # normalize到0,方便训练
    data = data - 1
    # lstm 读取的最大周期长度
    sequence_length = seq_len + 1

    # 设置窗口
    # 按周期窗口,添加数据到result
    windows_num = data.shape[0] + 1 - sequence_length
    index = np.arange(windows_num)[:, np.newaxis] + \
        np.arange(sequence_length)[np.newaxis, :]
    #index = index.reshape(windows_num * sequence_length)
    result = data[index]
    #result = result.reshape((windows_num, sequence_length))

    # 分割出训练和测试数据
    row = int(round(0.9 * result.shape[0]))
    train = result[:row]
    np.random.shuffle(train)
    # 取出窗口中最后一个周期最为y,其余作为x输入
    x_train = train[:, :-1]
    y_train = train[:, -1, 3]
    x_test = result[row:, :-1]
    y_test = result[row:, -1, 3]

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]
