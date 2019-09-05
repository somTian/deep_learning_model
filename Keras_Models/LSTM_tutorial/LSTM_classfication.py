from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Bidirectional,GRU,Dropout
from keras.preprocessing import sequence,text
from loaddata import readData,segSent,lablecode,datasplit


def dataPre(max_len,xtrain,xtest):
    # 使用 keras tokenizer
    '''
    文本标记实用类。
    该类允许使用两种方法向量化一个文本语料库： 将每个文本转化为一个整数序列（每个整数都是词典中标记的索引）；
    或者将其转化为一个向量，其中每个标记的系数可以是二进制值、词频、TF-IDF权重等。
    num_words: 需要保留的最大词数，基于词频。只有最常出现的 num_words 词会被保留。
    '''
    token = text.Tokenizer(num_words=None)

    token.fit_on_texts(list(xtrain) + list(xtest))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xtest_seq = token.texts_to_sequences(xtest)

    # 对文本序列进行zero填充
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

    word_index = token.word_index

    return xtrain_pad, xtest_pad, word_index

'''
units: 正整数，输出空间的维度。
activation: 要使用的激活函数 (详见 activations)。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
dropout: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
'''

def LstmModel(num_word, embed_dim, max_length, num_labels):
    model = Sequential()
    model.add(Embedding((num_word) + 1, embed_dim, input_length=max_length))
    model.add(LSTM(units=100, recurrent_dropout=0.1))   # rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_labels,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    return model

def BLstmModel(num_word, embed_dim, max_length, num_labels):
    model = Sequential()
    model.add(Embedding((num_word) + 1, embed_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(100, dropout=0.1)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_labels,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    return model


def DGRU(num_word, embed_dim,embedding_matrix, max_length, num_labels):
    model = Sequential()
    model.add(Embedding(num_word + 1,embed_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
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
    embed_dim = 100  # 可调节参数

    rawdata = readData(file)
    X = segSent(rawdata,sw_file)
    Y = lablecode(rawdata)
    num_labels = Y.shape[1]
    # max_len = max(len(x) for x in X)
    max_len = 200 #可调节参数
    # print(max_len)
    xtrain, xtest, ytrain, ytest = datasplit(X, Y)

    xtrain_pad, xtest_pad, word_index = dataPre(max_len,xtrain,xtest)
    num_word = len(word_index)

    model = BLstmModel(num_word, embed_dim, max_len, num_labels)
    model.fit(x=xtrain_pad, y=ytrain, batch_size=128, epochs=10, verbose=1, validation_data=(xtest_pad, ytest))
    '''
    verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    validation_data: ，用来评估损失，以及在每轮结束时的任何模型度量指标。模型将不会在这个数据上进行训练。
    '''

