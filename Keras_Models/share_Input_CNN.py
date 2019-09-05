from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate

'''
该模型输入采用大小为64×64像素的黑白图像。有两个CNN特征提取子模型共享该输入：第一个内核大小为4，第二个内核大小为8。
这些特征提取子模型的输出被平坦化为向量，并连接成一个长向量，并传递到完全连接的层，以便在最终输出层之前进行二进制分类。
'''

visible = Input(shape=(64,64,1))

conv1 = Conv2D(32,kernel_size=4,activation='relu')(visible)
pool1 = MaxPool2D(pool_size=(2,2))(conv1)
flat1 = Flatten()(pool1)

conv2 = Conv2D(16,kernel_size=8,activation='relu')(visible)
pool2 = MaxPool2D(pool_size=(2,2))(conv2)
flat2 = Flatten()(pool2)

merge = concatenate([flat1,flat2])

hidden = Dense(10,activation='relu')(merge)

output = Dense(1,activation='sigmoid')(hidden)

model = Model(inputs=visible,outputs=output)

print(model.summary())