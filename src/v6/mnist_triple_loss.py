import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.activations import softmax, tanh, relu
from keras.datasets import mnist
from keras import backend as keras
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam


FLAGS = None


def depict_loss_layer(args, FLAGS, alpha, prior):
    P = args
    N = keras.reshape(keras.sum(P, axis=0), (-1, 10))
    N = P / keras.pow(N, 0.5)
    D = keras.reshape(keras.sum(N, 1), (-1, 1))
    Q = N / D

    U = keras.variable(prior)
    F = keras.reshape(keras.mean(Q, axis=0), (-1, 10))

    C = Q * keras.log(Q / P)
    R = Q * keras.log(F / U)

    L = keras.reshape(keras.sum(C + alpha * R, axis=1), (-1, 1))

    return L

def build_triple_loss_model():
    # base model
    inputs = Input(shape=(28 * 28,))
    x = inputs

    encode_x = Dense(128, activation=tanh)(x)
    decode_x = Dense(28 * 28, activation=relu)(encode_x)
    softmax_x = Dense(10, activation=softmax)(encode_x)

    alpha = 1.0
    prior = [1 / float(10)] * 10
    loss_x = Lambda(depict_loss_layer, arguments={'FLAGS': FLAGS, 'alpha': alpha, 'prior': prior})(softmax_x)

    train_model = Model(inputs=[inputs], outputs=[decode_x, softmax_x, loss_x])
    infer_model = Model(inputs=[inputs], outputs=[softmax_x])

    return infer_model, train_model

infer_model, train_model = build_triple_loss_model()

infer_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])

loss_1 = mean_squared_error
loss_2 = categorical_crossentropy
loss_3 = lambda y_true, y_pred: y_pred
train_model.compile(optimizer=Adam(), loss=[loss_1, loss_2, loss_3], loss_weights=[1,1,1], metrics=[categorical_accuracy])

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
    # base_model.fit(x_train, y_train, verbose=0)
    # print(base_model.evaluate(x_test, y_test, verbose=0))

    train_model.fit(x_train, [x_train, y_train, y_train], verbose=0)
    print(train_model.evaluate(x_test, [x_test, y_test, y_test], verbose=0))


