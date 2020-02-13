# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 06:02:39 2020

@author: Andrei
"""
import numpy as np
import tensorflow as tf

def cnn_backbone(inputs, cnns):
  tf_inp = tf.keras.layers.Input(inputs)
  tf_x = tf_inp
  cnn_feats = []
  for conv in cnns:
    f = conv[0]
    k = conv[1]
    tf_x_cnn = tf.keras.layers.Conv1D(filters=f, kernel_size=k,
                                      padding='same')(tf_x)
    tf_x_cnn = tf.keras.layers.BatchNormalization()(tf_x_cnn)
    tf_x_cnn = tf.keras.layers.Activation('relu')(tf_x_cnn)
    cnn_feats.append(tf_x_cnn)
    
  return tf_inp, cnn_feats
    
def TextRCNN1(inputs, lstm_size=1024, cnns=[(32,1), (32,3), (64,5), (128,7)]):
  """
  slowest - uses a big LSTM on the concatenated grams
  """
  tf_input, cnn_feats = cnn_backbone(inputs=inputs, cnns=cnns)

  tf_x = tf.keras.layers.concatenate(cnn_feats)
  tf_x = tf.keras.layers.CuDNNLSTM(lstm_size)(tf_x)
  
  tf_readout = tf.keras.layers.Dense(1, activation='sigmoid')(tf_x)
  model = tf.keras.models.Model(tf_input, tf_readout, name='TextRCNN1')  
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model
  
def TextRCNN2(inputs, lstm_size=256, cnns=[(32,1), (32,3), (64,5), (128,7)]):
  """
  average - uses concatenated grams but also analyzes multiple lenghts by stacking RNNs
  """
  tf_input, cnn_feats = cnn_backbone(inputs=inputs, cnns=cnns)

  tf_x = tf.keras.layers.concatenate(cnn_feats)
  tf_x = tf.keras.layers.CuDNNLSTM(lstm_size, return_sequences=True)(tf_x)
  tf_x = tf.keras.layers.CuDNNLSTM(lstm_size, return_sequences=True)(tf_x)
  tf_x = tf.keras.layers.CuDNNLSTM(lstm_size, return_sequences=True)(tf_x)
  tf_x = tf.keras.layers.CuDNNLSTM(lstm_size)(tf_x)
  
  tf_readout = tf.keras.layers.Dense(1, activation='sigmoid')(tf_x)
  model = tf.keras.models.Model(tf_input, tf_readout, name='TextRCNN2')  
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model


def TextRCNN3(inputs, lstm_size=256, cnns=[(32,1), (32,3), (64,5), (128,7)]):
  """
  fast - here we analyze the intuition of having a rnn analyze a whole sequence of grams
         and only then concatenate multiple types of grams results
  """
  tf_input, cnn_feats = cnn_backbone(inputs=inputs, cnns=cnns)

  rcnns = []
  for cnn_feat in cnn_feats:
    tf_rcnn = tf.keras.layers.CuDNNLSTM(lstm_size)(cnn_feat)
    rcnns.append(tf_rcnn)
  tf_x = tf.keras.layers.concatenate(rcnns)
  
  tf_readout = tf.keras.layers.Dense(1, activation='sigmoid')(tf_x)
  model = tf.keras.models.Model(tf_input, tf_readout, name='TextRCNN3')  
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model


if __name__ == '__main__':
  inputs = (1500, 128)
  batch = 32
  obs = batch * 50
  
  X = np.random.rand(obs, inputs[0], inputs[1])
  y = np.random.randint(2, size=(obs,1))
  
  models = [TextRCNN1(inputs), TextRCNN2(inputs), TextRCNN3(inputs)]
  for model in models:
    model.summary()
    model.fit(X,y, batch_size=batch, epochs=3)
  