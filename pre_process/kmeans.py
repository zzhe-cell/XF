from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import jieba
import numpy as np
import matplotlib.cm as cm
import jieba.posseg as pseg
import paddle
from sklearn import metrics
paddle.enable_static()
jieba.enable_paddle()
import re

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=80, batch_size=50, random_state=20).fit(data).inertia_)
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

#停用词路径
stopword_path = "../data/stopword.txt"
#数据路径
filepath = "../data/信访事项（1）.csv"
data = pd.read_csv(filepath, encoding="GBK")
#加载停用词
stopword_list = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
#获取文本并去除里面的数字
texts = data["反映内容"].values.tolist()
for index, text in enumerate(texts):
    text = re.sub('[\d]', '', text)
    text = re.sub('[一|二|三|四|五|六|七|八|九|十|采油]+厂', '', text)
    texts[index] = text
#jieba分词
text_words = [list(jieba.cut(text)) for text in texts]
#将分词结果从列表转为空格拼接的字符串，方便用TfidfVectorizer处理
document = [" ".join(words) for words in text_words]
#获取文档中的人名，组织名和时间和地名作为额外的停用词
per_list, org_list, time_list, loc_list = get_other_stopword_list(text_words)
stopword_list = stopword_list + per_list + org_list + time_list + loc_list
#构建tfidf矩阵
tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.9, stop_words=stopword_list).fit(document)
tfidf_arr = tfidf_model.transform(document).toarray()
#用tfidf矩阵聚类，对比k值对sse的影响
# find_optimal_clusters(tfidf_arr, 24)
#选取k=16进行聚类，对参数进行网格搜索
init_sizes = [None, 50, 60, 70, 80]
batch_sizes = [30, 40, 50, 60, 1024]
n_inits = [3, 10, 20, 30, 50]
reassignment_ratios = [0.01, 0.02, 0.05, 0.1]
max_iters = [100, 200, 300, 400, 500]
best_score = 0.0
# for init_size in init_sizes:
#     for batch_size in batch_sizes:
#         for n_init in n_inits:
#             for reassignment_ratio in reassignment_ratios:
#                 for max_iter in max_iters:
#                     clusters = MiniBatchKMeans(n_clusters=16, init_size=init_size, batch_size=batch_size, n_init=n_init, reassignment_ratio=reassignment_ratio, max_iter=max_iter).fit_predict(tfidf_arr)
#                     sc = metrics.silhouette_score(tfidf_arr, clusters)
#                     if sc > best_score:
#                         best_score = sc
#                         best_parameters = {'init_size': init_size, 'batch_size': batch_size, 'n_init': n_init, 'reassignment_ratio': reassignment_ratio, 'max_iter': max_iter}
# print("best_score:{:.2f}".format(best_score))
# print("best_parameters:{}".format(best_parameters))
# init_size = best_parameters['init_size']
# batch_size = best_parameters['batch_size']
# n_init = best_parameters['n_init']
# reassignment_ratio = best_parameters['reassignment_ratio']
# max_iter = best_parameters['max_iter']
clusters = MiniBatchKMeans(n_clusters=16, init_size=80, batch_size=1024, n_init=20, reassignment_ratio=0.05, max_iter=100).fit_predict(tfidf_arr)
#将文本向量降维为2维并画出聚类结果
plot_pca(tfidf_arr, clusters)
#获取各类的前n个关键词
get_top_keywords(tfidf_arr, clusters, tfidf_model.get_feature_names(), 5)
#将聚类结果保存
res = get_result(data, clusters)
res.to_csv('../result/kmeans_result.csv', encoding='utf-8')

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