import tensorflow as tf
import numpy as np

class Network:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss='categorical_crossentropy', metrics=["accuracy"])
        return model
    
    def train(self, data, labels, val_data, val_labels):
        self.model.fit(data, labels, batch_size=32, epochs=5, validation_data=(val_data, val_labels))

    def save_model(path):
        self.model.save(path, save_format='h5')

    def load_model(path):
        self.model.load_model(path)

data = np.random.random((1000,4)).astype('float32')
labels = np.random.random((1000, 10)).astype('float32')

val_data = np.random.random((100,4)).astype('float32')
val_labels = np.random.random((100, 10)).astype('float32')

model = Network()

model.train(data, labels, val_data, val_labels)

class Agent:
    def __init__(self, number):
        self.id = number
    
    def make_action(self):
        data = np.random.random((1,4)).astype('float32')
        labels = np.random.random((1, 10)).astype('float32')
        return [data, labels]