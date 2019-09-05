from  keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 32


inputs = Input(shape=(784,))
encoder = Dense(encoding_dim, activation='relu')(inputs)
decoder = Dense(784, activation='sigmoid')(encoder)

autoencoder = Model(input=inputs, output=decoder)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))