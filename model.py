from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


def mymodel(vocab_size):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128,input_shape=(None,vocab_size),return_sequences=True))
    model.add(tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.3))
    model.add(tf.keras.layers.Dense(vocab_size,activation='softmax'))
    return model