import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import losses
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib


def AIAO(model, X_test, y_test, seq_len, ):

    print("mean test", np.mean(y_test),)

    pt = model.predict(X_test)[:, 0]
    print(pt.shape)

    print("mean test", np.mean(y_test), np.mean(pt))
    r = 1
    r0 = 1
    rall = 1
    for y, p in zip(y_test, pt):
        ratio = y + 1
        rall *= ratio
        if p > 0:
            r *= ratio
        if y > 0:
            r0 *= ratio
    print(r0, rall, r, )

    matplotlib.use('Qt5Agg')
    plt.plot(y_test)
    plt.plot(pt)
    plt.show()



