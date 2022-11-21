from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import jieba.posseg as pseg

from sklearn import metrics, manifold

from pre_process.preprocess import Preprocess

import re


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

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data).groupby(clusters).mean()

    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

def get_result(data, clusters):
    res = pd.DataFrame()
    types = list(set(clusters))
    type_list = []
    for type in types:
        for id, item in enumerate(clusters):
            if item == type:
                type_list.append(type)
                d = data.iloc[[id]]
                res = pd.concat([res, data.iloc[[id]]], ignore_index=True)
    res['类别'] = type_list
    return res
# def get_result(data, clusters, dic):
#     res = pd.DataFrame()
#     types = list(set(clusters))
#     type_list = []
#     for type in types:
#         for id, item in enumerate(clusters):
#             if item == type:
#                 type_list.append(type)
#                 d = data.iloc[[id]]
#                 res = pd.concat([res, data.iloc[[id]]], ignore_index=True)
#     res['类别'] = type_list
#     keyword_list = []
#     for i, row in res.iterrows():
#         keyword = dic[row['类别']]
#         keyword_list.append(keyword)
#     res['关键词'] = keyword_list
#     return res


def get_other_stopword_list(text_words, text_flags):
    per_list = []  # 人名列表
    org_list = []
    time_list = []
    loc_list = []
    for word_list, flag_list in zip(text_words, text_flags):
        for word, flag in zip(word_list, flag_list):
            if len(word) == 1:
                continue
            # words = pseg.cut(word, use_paddle=True)  # paddle模式
            # print(list(words))
            if flag == 'nr':
                if word not in per_list:
                    per_list.append(word)
            elif flag == 'nt':
                if word not in org_list:
                    org_list.append(word)
            elif flag == 't':
                if word not in time_list:
                    time_list.append(word)
            elif flag == 'ns':
                if word not in loc_list:
                    loc_list.append(word)
    return per_list, org_list, time_list, loc_list

stopword_path = "../data/stopword.txt"
#数据路径
filepath = "../data/信访事项（1）.csv"
data = pd.read_csv(filepath, encoding="GBK")
p = Preprocess(data)
data1 = p.preprocess()
arr1 = data1.values
#加载停用词
stopword_list = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
#获取文本并去除里面的数字
texts = data["反映内容"].values.tolist()
for index, text in enumerate(texts):
    text = re.sub('[\d]', '', text)
    text = re.sub('[一|二|三|四|五|六|七|八|九|十|采油]+厂', '', text)
    texts[index] = text
#jieba分词
text_words = []
text_flags = []
for text in texts:
    words = pseg.cut(text)
    w_list = [w.word for w in words]
    flag_list = [w.flag for w in words]
    text_words.append(w_list)
    text_flags.append(flag_list)
#将分词结果从列表转为空格拼接的字符串，方便用TfidfVectorizer处理
document = [" ".join(words) for words in text_words]
#获取文档中的人名，组织名和时间和地名作为额外的停用词
per_list, org_list, time_list, loc_list = get_other_stopword_list(text_words, text_flags)
stopword_list = stopword_list + per_list + org_list + time_list + loc_list
#构建tfidf矩阵
tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.9, stop_words=stopword_list).fit(document)
tfidf_arr = tfidf_model.transform(document).toarray()
# tSNE = manifold.TSNE(n_components=2, init='pca')
# pca = PCA(n_components=10).fit_transform(tfidf_arr)
train_arr = np.concatenate((arr1, tfidf_arr), axis=1)
# tfidf_arr1 = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(tfidf_arr)
#DBSCAN调参
# epss = [1.0449,1.045,1.0451,1.0452,1.0453]
# min_sampless = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# best_score = 0.0
# for eps in epss:
#     for min_samples in min_sampless:
#         Dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(train_arr)
#         clusters = Dbscan_model.labels_
#         if len(list(set(clusters))) < 2:
#             sc = 0
#         else:
#             sc = metrics.silhouette_score(train_arr, clusters)
#         if sc > best_score:
#             best_score = sc
#             best_parameters = {'eps': eps, 'min_samples': min_samples}
# print("best_score:{:.2f}".format(best_score))
# print("best_parameters:{}".format(best_parameters))
# eps = best_parameters['eps']
# min_samples = best_parameters['min_samples']

Dbscan_model1 = DBSCAN(eps=1.2, min_samples=1).fit(tfidf_arr)
clusters1 = Dbscan_model1.labels_
sc1 = metrics.silhouette_score(tfidf_arr, clusters1)
dbi1 = metrics.davies_bouldin_score(tfidf_arr, clusters1)

# best_score = 0.0
# for eps in epss:
#     for min_samples in min_sampless:
#         Dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(tfidf_arr)
#         clusters = Dbscan_model.labels_
#         if len(list(set(clusters))) < 2:
#             sc = 0
#         else:
#             sc = metrics.silhouette_score(tfidf_arr, clusters)
#         if sc > best_score:
#             best_score = sc
#             best_parameters = {'eps': eps, 'min_samples': min_samples}
# print("best_score:{:.2f}".format(best_score))
# print("best_parameters:{}".format(best_parameters))
# eps = best_parameters['eps']
# min_samples = best_parameters['min_samples']
#
# Dbscan_model2 = DBSCAN(eps=eps, min_samples=min_samples).fit(tfidf_arr)
# clusters2 = Dbscan_model2.labels_
# sc2 = metrics.silhouette_score(tfidf_arr, clusters2)
# dbi2 = metrics.davies_bouldin_score(tfidf_arr, clusters2)
print("sc1:{:.2f}".format(sc1))
# print("sc2:{:.2f}".format(sc2))
print("dbi1:{:.2f}".format(dbi1))
# print("dbi2:{:.2f}".format(dbi2))
#将文本向量降维为2维并画出聚类结果
# fig, axs = plt.subplots(1, 2, figsize=(8,8))
# labels1 = [cm.hsv(i / max(clusters1)) for i in clusters1]
# labels2 = [cm.hsv(i / max(clusters2)) for i in clusters2]
# axs[0].scatter(train_arr1[:, 0], train_arr1[:, 1], c=labels1, s=30)
# axs[1].scatter(tfidf_arr1[:, 0], tfidf_arr1[:, 1], c=labels2, s=30)
# plt.show()
# #获取各类的前n个关键词
# keywords = get_top_keywords(tfidf_arr, clusters, tfidf_model.get_feature_names(), 5)
#将聚类结果保存
res = get_result(data, clusters1)
res.to_csv('../result/dbscan_result.csv', encoding='utf-8')

#print(tfidf_arr)
# for i in range(len(tfidf_arr)):
#     dis = []
#     for j in range(len(tfidf_arr)):
#         dis.append(distance.cosine(tfidf_arr[i], tfidf_arr[j]))
#
#     dis_sort = np.argsort(dis)
#     slice = dis_sort[:10]
#     jinsi = data.iloc[slice, :]
#     jinsi.to_csv('../data/第{}条数据的10近邻_文本.csv'.format(i), encoding='utf-8')