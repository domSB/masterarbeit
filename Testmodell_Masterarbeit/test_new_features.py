import tensorflow as tf
from threading import Thread
import numpy as np

class Network:
    def __init__(self, name):
        self.model = self.create_model(name)

    def create_model(self, name):
        with tf.name_scope(name):
            inputs = tf.keras.Input(shape=(4,))
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss='categorical_crossentropy', metrics=["accuracy"])
            return model
    
    def train(self, data, labels, session, id):
        with session.as_default(), session.graph.as_default(), tf.name_scope(id):
            self.model.fit(data, labels, batch_size=32, epochs=5)

    def save_model(path):
        self.model.save(path, save_format='h5')

    def load_model(path):
        self.model.load_model(path)

class Agent:
    def __init__(self, number):
        self.id = number
    
    def make_action(self):
        data = np.random.random((1000,4)).astype('float32')
        labels = np.random.random((1000, 10)).astype('float32')
        return [data, labels]



def func(agent, id, session):
    with tf.Session(graph=tf.Graph()) as session:
        with tf.name_scope(id):
            net =Network(id)
            data, labels = agent.make_action()
            print(data[0], labels[0])
            val_data, val_labels = agent.make_action()
            print(type(data), type(labels))
            net.train(data, labels, session, id)
            print(id, " is done")
            return



if __name__ == '__main__':
#    g = tf.Graph()
#    session = tf.Session(graph=g)
#    with g.as_default(), session.as_default():
#    tf.keras.backend.set_session(session)
    coord = tf.train.Coordinator()
    net1 = Network("1")
    net2 = Network("2")
    agent1 = Agent("1")
    agent2 = Agent("2")

    t1 = Thread(None, func, args=(agent1, "1", None))
    t2 = Thread(None, func, args=(agent2, "2", None))
    t1.start()
    t2.start()
    coord.join([t1, t2])
