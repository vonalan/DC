import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.activations import softmax, tanh
from keras.datasets import mnist
from keras import backend as keras
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam


FLAGS = None

def depict_loss_layer(args, FLAGS, alpha, prior):
    P = args
    N = keras.reshape(keras.sum(P, axis=0), (-1, 10))
    N = P / keras.pow(N, 0.5)
    D = keras.reshape(keras.sum(N, 1), (-1, 1))
    Q = N / D

    C = Q * keras.log(P) * -1
    L = keras.reshape(keras.sum(C, axis=1), (-1, 1))
    return L

# base model
inputs = Input(shape=(28 * 28, ))
x = inputs
x = Dense(128, activation=tanh)(x)
x = Dense(10, activation=softmax)(x)
outputs = x
base_model = Model(inputs, outputs)

alpha = 1.0
prior = None
depict_loss = Lambda(depict_loss_layer, arguments={'FLAGS': FLAGS, 'alpha': alpha, 'prior': prior})(outputs)
train_model = Model(inputs=[inputs], outputs=[outputs, depict_loss])

base_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])

loss_1 = categorical_crossentropy
loss_2 = lambda y_true, y_pred: y_pred
train_model.compile(optimizer=Adam(), loss=[loss_1, loss_2], loss_weights=[1,0], metrics=[categorical_accuracy])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = np.reshape(x_train, (-1, 28*28))
temp = np.zeros((y_train.shape[0], 10))
for i in range(temp.shape[0]):
    temp[i,y_train[i]] = 1
y_train = temp

x_test = np.reshape(x_test, (-1, 28*28))
temp = np.zeros((y_test.shape[0], 10))
for i in range(temp.shape[0]):
    temp[i,y_test[i]] = 1
y_test = temp

for i in range(20):
    base_model.fit(x_train, y_train, verbose=0)
    print(base_model.evaluate(x_test, y_test, verbose=0))
    #
    # train_model.fit(x_train, [y_train, y_train], verbose=0)
    # print(train_model.evaluate(x_test, [y_test, y_test], verbose=0))


