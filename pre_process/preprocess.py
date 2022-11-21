# -*- coding: utf-8 -*-
import pandas as pd
# import xmnlp
import time
#from xmnlp.sv import SentenceVector
import numpy as np
#xmnlp.set_model('../xmnlp-onnx-models')

class Preprocess:
    def __init__(self, data):
        self.data = data
    def preprocess(self):
        data1 = self.data.drop(columns=['登记单位', '信访件编号','来访人','人数','来访诉求','信访件状态'])
        data1['来访时间'] = data1['来访时间'].astype(str)
        data1['来访时间'] = data1['来访时间'].apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d')))
        data1 = pd.get_dummies(data1, columns=['信访目的', '涉事单位'])
        #lists = data1["反映内容"].values.tolist()
        #sv = SentenceVector(genre="通用")
        #for i in range(len(lists)):
            #lists[i] = sv.transform(lists[i])
        #data1['文本向量'] = lists
        data1.drop('反映内容', axis=1, inplace=True)
        #除了文本列之外的列进行min-max归一化处理
        for i in list(data1.columns):
            # 获取各个指标的最大值和最小值
            if i == '反应内容':
                continue
            Max = np.max(data1[i])
            Min = np.min(data1[i])
            data1[i] = (data1[i] - Min) / (Max - Min)
        return data1
