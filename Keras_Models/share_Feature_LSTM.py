from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate

visible = Input(shape=(100,1))

extract1 = LSTM(10)(visible)

interp1 = Dense(10,activation='relu')(extract1)

interp11 = Dense(10,activation='relu')(extract1)
interp12 = Dense(20,activation='relu')(interp11)
interp13 = Dense(20,activation='relu')(interp12)

merge = concatenate([interp1,interp13])

output = Dense(1,activation='sigmoid')(merge)

model = Model(inputs=visible,outputs=output)

print(model.summary())