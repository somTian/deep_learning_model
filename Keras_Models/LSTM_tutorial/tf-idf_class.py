from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import jieba
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# 1、读取数据
def readData(file):
    rawdata = pd.read_excel(file,sheetname='sheet1')
    return rawdata

# 2、中文文本分词、去停词
def segSent(data):
    """
    中文数据集请使用这个清理文本内容
    """
    data['分词'] = data['文本'].apply(lambda i:jieba.lcut(i))
    data['分词'] = [' '.join(i) for i in data['分词']]
    return data

# 2、英文文本清理标点及非英字符
def clean_str(string):
    """
    英文数据集请用这个清理文本内容
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 4、 标签向量化
def lablecode(data):
    y = LabelEncoder().fit_transform(data['类别'].values)
    return y

# 5、划分训练集测试集
def datasplit(data,y):
    xtrain, xvalid, ytrain, yvalid = train_test_split(data, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)
    return xtrain, xvalid, ytrain, yvalid

def tfidfvec(data,sw_file):
    stwlist = [line.strip() for line in open(sw_file,'r', encoding='utf-8').readlines()]
    tfv = TfidfVectorizer(stop_words=stwlist, max_df=1.0, min_df=1,use_idf=True, smooth_idf=True)
    tfidf_vec = tfv.fit_transform(data)
    return tfidf_vec

if __name__ == '__main__':

    file = 'data/nlpdata.xlsx'
    sw_file = 'data/stopwords.txt'

    rawdata = readData(file)
    X = segSent(rawdata[:2000])
    Y = lablecode(rawdata[:2000])
    tf_vec = tfidfvec(X['分词'][:2000],sw_file)

    xtrain, xtest, ytrain, ytest = datasplit(tf_vec,Y)

    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    predict = clf.predict(xtest)
    print(accuracy_score(ytest,predict))
