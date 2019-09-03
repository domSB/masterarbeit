import numpy as np
import tensorflow as tf

params = {
    'time_steps': 2,
    'dynamic_state_shape': 5,
    'static_state_shape': 3,
    'forecast_state': 3,
    'learning_rate': 0.001,
    'batch_size': 10
}

dynamic_input, static_input, label = (
    np.array(
        [np.random.sample((100, 2, 5))]
    ),
    np.array(
        [np.random.sample((100, 3))]
    ),
    np.array(
        [np.random.sample((100, 1))]
    )
)

# dataset = tf.data.Dataset.from_tensor_slices((dynamic_input, static_input, label)).batch(params['batch_size'])

# iterator = dataset.make_one_shot_iterator()


def generator(_dynamic_input, _static_input, _label):
    global params
    bs = params['batch_size']
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'dynamic_input': _dynamic_input[0], 'static_input': _static_input[0]}, _label[0])
    )
    dataset = dataset.batch(bs)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def generator_single(_dynamic_input, _label):
    global params
    bs = params['batch_size']
    dataset = tf.data.Dataset.from_tensor_slices(({'dynamic_input': _dynamic_input[0]}, _label[0]))
    dataset = dataset.batch(bs)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def build_double_input_model():
    global params
    dynamic_inputs = tf.keras.Input(shape=(params['time_steps'], params['dynamic_state_shape']), name='dynamic_input')
    static_inputs = tf.keras.Input(shape=(params['static_state_shape'], ), name='static_input')
    dynamic_x = tf.keras.layers.LSTM(
        10,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="LSTM_1",
        return_sequences=True
    )(dynamic_inputs)
    dynamic_x = tf.keras.layers.LSTM(
        10,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="LSTM_2"
    )(dynamic_x)
    x = tf.keras.layers.concatenate([dynamic_x, static_inputs])
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_1"
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_2"
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_3"
    )(x)
    predictions = tf.keras.layers.Dense(params['forecast_state'], activation='relu', name="predictions")(x)
    _model = tf.keras.Model(inputs=[dynamic_inputs, static_inputs], outputs=predictions)
    rms = tf.keras.optimizers.RMSprop(lr=params['forecast_state'])
    _model.compile(optimizer=rms, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
    return _model


def build_single_input_model():
    global params
    dynamic_inputs = tf.keras.Input(shape=(params['time_steps'], params['dynamic_state_shape']), name='dynamic_input')
    dynamic_x = tf.keras.layers.LSTM(
        10,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="LSTM_1",
        return_sequences=True
    )(dynamic_inputs)
    dynamic_x = tf.keras.layers.LSTM(
        10,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="LSTM_2"
    )(dynamic_x)
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_1"
    )(dynamic_x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_2"
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="Dense_3"
    )(x)
    predictions = tf.keras.layers.Dense(params['forecast_state'], activation='relu', name="predictions")(x)
    _model = tf.keras.Model(inputs=dynamic_inputs, outputs=predictions)
    rms = tf.keras.optimizers.RMSprop(lr=params['forecast_state'])
    _model.compile(optimizer=rms, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
    return _model


model = build_double_input_model()
single = build_single_input_model()

training_set = generator_single(dynamic_input, label)

multiple_training_set = generator(dynamic_input, static_input, label)

single.fit(training_set.make_one_shot_iterator(), steps_per_epoch=10, epochs=1)


model.fit(multiple_training_set.make_one_shot_iterator(), steps_per_epoch=10, epochs=10)



