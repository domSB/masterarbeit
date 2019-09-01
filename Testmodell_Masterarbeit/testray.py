import ray
# import tensorflow as tf

import keras
import numpy as np


def get_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_shape=((13, ))))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    sgd = keras.optimizers.SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['acc'])
    return model


@ray.remote
class ParameterServer(object):
    def __init__(self, values):
        self.values = [value.copy() for value in values]
        self.weigths = values

    def push(self, values):
        self.values = [value.copy() for value in values]

    def pull(self):
        return self.weigths


@ray.remote
class Worker(object):
    def __init__(self, ps):
        self.model = get_model()
        self.ps = ps
        weights = self.model.get_weights()
        update_weigths = ray.get(ps.pull.remote())
        for i in range(len(weights)):
            weights[i] = update_weigths[i]
        self.model.set_weights(weights)

    def update_weigths(self):
        self.model.set_weights(weights)

    def feed_forward(self):
        state = np.zeros((1, 13))
        ergebnis = self.model.predict(state)
        return ergebnis


(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
ray.init()
bigboss = get_model()
hist = bigboss.fit(x_train, y_train, epochs=20, batch_size=64)
init_weigths = bigboss.get_weights()
ps = ParameterServer.remote(init_weigths)
workers = [Worker.remote(ps) for i in range(3)]
for i in range(3):
    [worker.update_weigths.remote() for worker in workers]
    print(ray.get([worker.feed_forward.remote() for worker in workers]))
    hist = bigboss.fit(x_train, y_train, epochs=25, batch_size=64, verbose=0)
    update_weigths = bigboss.get_weights()
    upload_id = ps.push.remote(update_weigths)
    print(bigboss.predict(np.zeros((1, 13))))



