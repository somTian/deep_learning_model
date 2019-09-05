from numpy import random

Matrix_RANK = 1005
v = random.randint(0,1,size=(Matrix_RANK, Matrix_RANK))  #105*105

with open('data/email-Eu-core2.txt', 'r', encoding="utf-8") as f:
   for line in f:
      line_split = line.strip().replace(' ','').split(',')
      a = int(line_split[0])
      b = int(line_split[1])
      v[a,a] = 1
      v[b,b] = 1
      v[a,b] = 1
      v[b, a] = 1

print(v)

print(v.shape)

from sklearn.model_selection import train_test_split

train, test = train_test_split(v, test_size=0.1, random_state=42)


from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 50



from keras import regularizers

inputs = Input(shape=(Matrix_RANK,))
# add a Dense layer with a L1 activity regularizer
encoder = Dense(encoding_dim, activation='tanh',
                activity_regularizer=regularizers.l1(10e-5))(inputs)
decoder = Dense(Matrix_RANK, activation='sigmoid')(encoder)

autoencoder_reg = Model(input=inputs, output=decoder)

print(autoencoder_reg.summary())

autoencoder_reg.compile(optimizer='adadelta', loss='binary_crossentropy')

pred_inputs = Input(shape=(Matrix_RANK,))
pred_encoder = Dense(encoding_dim, weights=autoencoder_reg.layers[1].get_weights(), activation='tanh')(pred_inputs)
pred_model =  Model(input=pred_inputs, output=pred_encoder)

autoencoder_reg.fit(train, train,
                nb_epoch=3000,
                shuffle=True,
                validation_data=(test,test)
                )


encoded_data = pred_model.predict(v)
import numpy
numpy.savetxt("50_email-Eu-core.txt", encoded_data)
print(encoded_data)