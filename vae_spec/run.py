#!/usr/bin/env python

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split


rs=0
batch_size = 32
epochs = 10000
xscale=1.0e5

data=np.load('/home/ichinohe/scratch/spec/data/data_1t_1890.npz')
xdata=data['xdata']/xscale
ydata=data['ydata']
xtrain,xtest,_,_ = train_test_split(xdata,ydata,test_size=0.2,random_state=rs)

orig_dim = xdata.shape[1]
nnode = 128
nlatent = 8

## encoder
x = inputs = Input(shape=(orig_dim,), name='input')
x = Dense(nnode, activation='relu')(x)
x = Dense(nnode, activation='relu')(x)
x = Dense(nnode, activation='relu')(x)

zm = Dense(nlatent, name='mu')(x)
zs = Dense(nlatent, name='ln_s2')(x)

def sampling(args):
    zm,zs = args
    batch = K.shape(zm)[0]
    dim = K.int_shape(zm)[1]
    epsilon = K.random_normal(shape=(batch,dim))
    return zm + K.exp(0.5*zs)*epsilon

z = Lambda(sampling, output_shape=(nlatent,), name='z')([zm,zs])

encoder = Model(inputs=inputs, outputs=[zm,zs,z], name='encoder')

## decoder
x = decoder_inputs = Input(shape=(nlatent,), name='decoder_input')
x = Dense(nnode, activation='relu')(x)
x = Dense(nnode, activation='relu')(x)
x = Dense(nnode, activation='relu')(x)
outputs = Dense(orig_dim, activation='softplus')(x)

decoder = Model(inputs=decoder_inputs, outputs=outputs, name='decoder')

## vae
vae_outputs = decoder(encoder(inputs)[2])
vae = Model(inputs=inputs, outputs=vae_outputs, name='vae')

## summary
encoder.summary()
decoder.summary()
vae.summary()

### optimization
x0=inputs*xscale
y0=vae_outputs*xscale

def log_poisson(t,p):
    return K.mean(p - t*K.log(p+K.epsilon()) + t*K.log(t+K.epsilon()) - t + K.log(t+K.epsilon())/2.0, axis=-1)

reconstruction_loss = orig_dim * log_poisson(x0,y0)
kl_loss = -0.5 * K.sum(1+zs-K.square(zm)-K.exp(zs), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(loss=None, optimizer='adam', metrics=[])

### training
savebest = ModelCheckpoint(filepath="./dump_weights_best.h5",
                           verbose=1, monitor='val_loss',
                           save_best_only=True, period=1)

history = vae.fit(xtrain,
                  batch_size=batch_size,epochs=epochs,
                  validation_data=(xtest,None),
                  verbose=1,
                  callbacks=[savebest])

### save
vae.save('dump_model.h5')
np.savez('dump_history',
         epoch=history.epoch,
         history=history.history)
