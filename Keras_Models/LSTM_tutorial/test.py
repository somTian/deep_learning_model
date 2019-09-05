from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense

model = Sequential()
model.add(Embedding(10, 5, input_length=8))
model.add(LSTM(20))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

