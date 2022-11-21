import numpy as np
from matplotlib import cm, pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA

from distance import Distance
from preprocess import Preprocess
import pandas as pd
from sklearn.cluster import DBSCAN

def plot_pca(data, labels):
    data = np.array(data)
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

filepath = '../data/信访事项（1）.csv'
data = pd.read_csv(filepath, encoding="GBK")
p = Preprocess(data)
data1 = p.preprocess()
arr = data1['文本向量'].values
d = Distance(data1)
dis_matrix = d.get_dis_matrix(weights=[10, 1, 1, 5])
epss = [3, 4, 5, 6, 7]
min_sampless = [6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20]
best_score = 0.0
for eps in epss:
    for min_samples in min_sampless:
        Dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dis_matrix)
        clusters = Dbscan_model.labels_
        if len(list(set(clusters))) < 2:
            sc = 0
        else:
            sc = metrics.silhouette_score(arr.tolist(), clusters)
        if sc > best_score:
            best_score = sc
            best_parameters = {'eps': eps, 'min_samples': min_samples}
print("best_score:{:.2f}".format(best_score))
print("best_parameters:{}".format(best_parameters))
eps = best_parameters['eps']
min_samples = best_parameters['min_samples']
Dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dis_matrix)
clusters = Dbscan_model.labels_
plot_pca(arr.tolist(), clusters)
res = get_result(data, clusters)
res.to_csv('../result/dbscan_result.csv', encoding='utf-8')
