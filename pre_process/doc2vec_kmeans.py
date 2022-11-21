import numpy as np
import pandas as pd
import re
import jieba
from pprint import pprint
import jieba.posseg as pseg
import os
import sys
from preprocess import Preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm

sys.stderr = open(os.devnull, "w")  # silence stderr
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import paddle

# Plotting tools
import matplotlib.pyplot as plt
def get_other_stopword_list(text_words):
    per_list = []  # 人名列表
    org_list = []
    time_list = []
    loc_list = []
    for word_list in text_words:
        for word in word_list:
            if len(word) == 1:
                continue
            words = pseg.cut(word, use_paddle=True)  # paddle模式
            # print(list(words))
            word, flag = list(words)[0]
            if flag == 'PER':
                if word not in per_list:
                    per_list.append(word)
            elif flag == 'ORG':
                if word not in org_list:
                    org_list.append(word)
            elif flag == 'TIME':
                if word not in time_list:
                    time_list.append(word)
            elif flag == 'LOC':
                if word not in loc_list:
                    loc_list.append(word)

    # print(per_list)
    return per_list, org_list, time_list, loc_list

def get_stop_words(path):
    return [item.strip() for item in open(path, 'r', encoding='utf-8').readlines()]

def drop_stopwords(line, stopwords):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()

def plot_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=99, replace=False)
    pca = PCA(n_components=2).fit_transform(data[max_items, :])
    idx = np.random.choice(range(pca.shape[0]), size=99, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]
    f, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax.set_title('PCA Cluster Plot')
    plt.show()
if __name__ == '__main__':
    paddle.enable_static()
    jieba.enable_paddle()
    sys.stderr = sys.__stderr__  # unsilence stderr
    stopword_path = "../data/stopword.txt"
    filepath = "../data/信访事项（1）.csv"
    data = pd.read_csv(filepath, encoding="GBK")
    texts = data["反映内容"].values.tolist()
    #去掉身份证号
    for index, text in enumerate(texts):
        text = re.sub('[\d|x]{3}', '', text)
        text = re.sub('[一|二|三|四|五|六|七|八|九|十|采油]+厂', '', text)
        texts[index] = text
    train_seg_text = [jieba.lcut(s) for s in texts]
    stopwords = get_stop_words(stopword_path)
    per_list, org_list, time_list, loc_list = get_other_stopword_list(train_seg_text)
    stopwords = list(set(stopwords + per_list + org_list + time_list + loc_list))
    train_st_text = [drop_stopwords(s, stopwords) for s in train_seg_text]
    print(train_st_text)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_st_text)]
    model = Doc2Vec(documents, vector_size=50, window=2, min_count=3)
    model.train(documents, total_examples=model.corpus_count, epochs=1000)
    infered_vectors_list = []
    for text, label in documents:
        vector = model.infer_vector(text)
        infered_vectors_list.append(vector)

    km = KMeans(n_clusters=34)
    clusters = km.fit_predict(infered_vectors_list)
    # print(km.labels_)
    sc = metrics.silhouette_score(infered_vectors_list, clusters)
    print('sc: ', sc)
    p = Preprocess(data)
    data1 = p.preprocess()
    # print(data1)
    data1 = data1.drop(columns='文本向量')
    data1 = data1.values
    data2 = np.array(infered_vectors_list)
    # print(data2)
    # for i, v in enumerate(data2):
    #     data2[i] = normalize(v)
    # print(data2)
    train_data = np.concatenate((data1, data2), axis=1)
    # find_optimal_clusters(infered_vectors_list, 40)
    # find_optimal_clusters(train_data, 40)
    km = KMeans(n_clusters=38)
    clusters = km.fit_predict(train_data)
    # print(km.labels_)
    sc = metrics.silhouette_score(infered_vectors_list, clusters)
    print('sc: ', sc)
    plot_pca(train_data, clusters)



