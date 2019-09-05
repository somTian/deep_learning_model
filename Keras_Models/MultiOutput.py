from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
'''
LSTM层解释输入序列，并返回每个时间步长的隐藏状态。
第一个输出模型创建一个堆栈的LSTM，解释特征，并进行二进制预测。
第二个输出模型使用相同的输出层对每个输入时间步长进行实值预测。
'''

visible = Input(shape=(100,1))
extract = LSTM(10,return_sequences=True)(visible)

class11 = LSTM(10)(extract)
class12 = Dense(10,activation='relu')(class11)

output1 = Dense(1,activation='sigmoid')(class12)

output2 = TimeDistributed(Dense(1,activation='linear'))(extract)

model = Model(inputs=visible,outputs=[output1,output2])

print(model.summary())