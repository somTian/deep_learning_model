from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Bidirectional,GRU,Dropout
from keras.preprocessing import sequence,text
from loaddata import readData,segSent,lablecode,datasplit


def dataPre(max_len,xtrain,xtest):
    # 使用 keras tokenizer
    token = text.Tokenizer(num_words=None)

    token.fit_on_texts(list(xtrain) + list(xtest))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xtest_seq = token.texts_to_sequences(xtest)

    # 对文本序列进行zero填充
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

    word_index = token.word_index

    return xtrain_pad, xtest_pad, word_index

def LstmModel(num_word, embed_dim, max_length):
    model = Sequential()
    model.add(Embedding((num_word) + 1, embed_dim, input_length=max_length))
    model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse',metrics=['mse'])
    return model

def BLstmModel(num_word, embed_dim, max_length):
    model = Sequential()
    model.add(Embedding((num_word) + 1, embed_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(100, dropout=0.1)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse',metrics=['mse'])
    return model

def DGRU(num_word, embed_dim,embedding_matrix, max_length):
    model = Sequential()
    model.add(Embedding(num_word + 1,embed_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse',metrics=['mse'])
    return model

if __name__ == '__main__':
    file = 'data/nlpdata.xlsx'
    sw_file = 'data/stopwords.txt'
    embed_dim = 100

    rawdata = readData(file)
    X = segSent(rawdata,sw_file)
    # Y = lablecode(rawdata)
    Y = rawdata['得分']
    # max_len = max(len(x) for x in X)
    max_len = 200
    # print(max_len)
    xtrain, xtest, ytrain, ytest = datasplit(X, Y)

    xtrain_pad, xtest_pad, word_index = dataPre(max_len,xtrain,xtest)
    num_word = len(word_index)


    model = BLstmModel(num_word, embed_dim, max_len)
    model.fit(xtrain_pad, ytrain, batch_size=128, epochs=10, verbose=1, validation_data=(xtest_pad, ytest))
