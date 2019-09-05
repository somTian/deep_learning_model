import pandas as pd
import jieba
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# 1、读取数据
def readData(file):
    rawdata = pd.read_excel(file,sheetname='sheet1')
    # print(rawdata.head())  # 输出前5行
    # print(rawdata.info())  # 数据信息
    # print(rawdata['类别'].unique())
    return rawdata


# 2、中文文本分词、去停词
def segSent(data,stopwords_file):
    """
    中文数据集请使用这个清理文本内容
    """
    data['分词'] = data['文本'].apply(lambda i:jieba.lcut(i))
    # print(data.head())
    stwlist = [line.strip() for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
    new_data = []
    for sent in data['分词'].values:
        words = [w for w in sent if not w in stwlist]
        new_data.append(words)

    # print(new_data[:5])

    return new_data


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
    y = LabelBinarizer().fit_transform(data['类别'].values)
    # print(y)
    return y

# 5、划分训练集测试集
def datasplit(data,y):
    xtrain, xvalid, ytrain, yvalid = train_test_split(data, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)
    return xtrain, xvalid, ytrain, yvalid

