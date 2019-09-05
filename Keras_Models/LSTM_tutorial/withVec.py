import numpy as np
from keras.models import Sequential
from keras.layers import Embedding,Dense,GRU,Dropout
from keras.preprocessing import sequence,text
from loaddata import readData,segSent,lablecode,datasplit

def embed_mat(vec_file):
    embeddings_index = {}
    f = open(vec_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


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

def reconstructVec(word_index,embeddings_index,e):

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def DGRU(num_word, embed_dim,embedding_matrix, max_len, num_labels):
    model = Sequential()
    model.add(Embedding(num_word + 1,embed_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(GRU(100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(100, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_labels,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    return model

if __name__ == '__main__':
    file = 'data/nlpdata.xlsx'
    sw_file = 'data/stopwords.txt'
    vec_file = 'data/glove.840B.300d.txt'
    embed_dim = 100

    embeddings_index = embed_mat(vec_file)

    rawdata = readData(file)
    X = segSent(rawdata,sw_file)
    Y = lablecode(rawdata)
    num_labels = Y.shape[1]
    # max_len = max(len(x) for x in X)
    max_len = 200
    # print(max_len)
    xtrain, xtest, ytrain, ytest = datasplit(X, Y)

    xtrain_pad, xtest_pad, word_index = dataPre(max_len,xtrain,xtest)
    num_word = len(word_index)
    embedding_matrix = reconstructVec(word_index,embeddings_index,embed_dim)

    model = DGRU(num_word, embed_dim,embedding_matrix, max_len, num_labels)
    model.fit(xtrain_pad, ytrain, batch_size=128, epochs=10, verbose=1, validation_data=(xtest_pad, ytest))

    print(model.predict(xtest))