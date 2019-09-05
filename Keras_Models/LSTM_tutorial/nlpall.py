import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence, text

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data=pd.read_excel('/home/kesci/input/Chinese_NLP6474/复旦大学中文文本分类语料.xlsx','sheet1')

import jieba
jieba.enable_parallel() #并行分词开启
data['文本分词'] = data['正文'].apply(lambda i:jieba.cut(i) )
data['文本分词'] =[' '.join(i) for i in data['文本分词']]

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.分类.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)


def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

stwlist=[line.strip() for line in open('/home/kesci/input/stopwords7085/停用词汇总.txt',
'r',encoding='utf-8').readlines()]
tfv = NumberNormalizingVectorizer(min_df=3,
                                  max_df=0.5,
                                  max_features=None,
                                  ngram_range=(1, 2),
                                  use_idf=True,
                                  smooth_idf=True,
                                  stop_words = stwlist)

# 使用TF-IDF来fit训练集和测试集（半监督学习）
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)

#利用提取的TFIDF特征来fit一个简单的Logistic Regression
clf = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)


# 词向量（Word Vectors）

X=data['文本分词']
X=[i.split() for i in X]

def trainWordVec():
    import gensim
    model = gensim.models.Word2Vec(X,min_count =5,window =8,size=100)   # X是经分词后的文本构成的list，也就是tokens的列表的列表
    embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

    print('Found %s word vectors.' % len(embeddings_index))


# 使用 keras tokenizer
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#对文本序列进行zero填充
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

def sent2vec(s):
    import jieba
    jieba.enable_parallel() #并行分词开启
    words = str(s).lower()
    #words = word_tokenize(words)
    words = jieba.lcut(words)
    words = [w for w in words if not w in stwlist]
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            #M.append(embeddings_index[w])
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# 对训练集和验证集使用上述函数，进行文本向量化处理
xtrain_w2v = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_w2v = [sent2vec(x) for x in tqdm(xvalid)]


# 在使用神经网络前，对数据进行缩放
scl = preprocessing.StandardScaler()
xtrain_w2v_scl = scl.fit_transform(xtrain_w2v)
xvalid_w2v_scl = scl.transform(xvalid_w2v)

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#在模型拟合时，使用early stopping这个回调函数（Callback Function）
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100,
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])