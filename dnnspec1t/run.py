#!/usr/bin/env python

import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

width = 64
depth = 3
rs = 1337 + 0
np.random.seed(rs)

batch_size = 50
epochs = 1000
ntrain = 9000

x_scale = 1.0e5
y0 = (1.0, 10.0)
y1 = (0.1, 1.5)
y2 = (0.0, 0.1)
y3 = (-2.0, 0.0)

###

data = np.load('data_1t_1890.npz')
X_data = data['X_data']
Y_data = data['Y_data']

np.random.seed(rs)
np.random.shuffle(X_data)
np.random.seed(rs)
np.random.shuffle(Y_data)

###

X_data = X_data/x_scale

_yd = Y_data.T
_yd[0] = (_yd[0]           -y0[0])/(y0[1]-y0[0])
_yd[1] = (_yd[1]           -y1[0])/(y1[1]-y1[0])
_yd[2] = (_yd[2]           -y2[0])/(y2[1]-y2[0])
_yd[3] = (np.log10(_yd[3]) -y3[0])/(y3[1]-y3[0])
Y_data = _yd.T

###

X_train = X_data[:ntrain]
X_test  = X_data[ntrain:]
Y_train = Y_data[:ntrain]
Y_test  = Y_data[ntrain:]

###

x = inputs = Input(shape=(X_data.shape[1],))
for i in range(depth):
    x = Dense(width, activation='relu')(x)

outputs = Dense(4)(x)
model = Model(inputs=inputs, outputs=outputs)

model.summary()
model.compile(loss='mse', optimizer='Adam')

###

savebest = ModelCheckpoint(filepath="./dump_weights_best.h5",
                           verbose=1, monitor='val_loss',
                           save_best_only=True, period=10)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    callbacks=[savebest])

###

model.save('dump_model.h5')
np.savez('dump_history',
         epoch=history.epoch,
         history=history.history)
