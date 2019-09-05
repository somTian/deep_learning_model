from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D

visible = Input(shape=(64,64,1))
conv1 = Conv2D(32,kernel_size=4,activation='relu')(visible)
pool1 = MaxPool2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(16,kernel_size=4,activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2,2))(conv2)
hidden = Dense(10,activation='relu')(pool2)
output = Dense(1,activation='sigmoid')(hidden)
model = Model(inputs=visible,outputs=output)

print(model.summary())
