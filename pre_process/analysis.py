# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
filepath = "../data/信访事项统计表明细_xx.csv"
data = pd.read_csv(filepath, encoding="GBK")
data['来访时间'] = pd.to_datetime(data['来访时间']).apply(lambda x : x.strftime("%Y-%m-%d"))
# data = data.sort_values(by='来访时间')
# category_num = len(data["信访目的"].value_counts().index)
# category_count = data["信访目的"].value_counts()
# orgnaization_num = len(data["涉事单位"].value_counts().index)
# orgnaization_count = data["涉事单位"].value_counts()
# data["涉事单位"].value_counts().plot.bar()
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.subplots_adjust(bottom=0.5)
# plt.show()
fanyings = []
suqiu = data['来访诉求']
for text in suqiu:
    pattern = re.compile(r'反映[\s\S]+。')
    fanying = re.search(pattern, text)
    if fanying is None:
        fanying = ""
    else:
        fanying = fanying.group(0)
    fanyings.append(fanying)
data['反映内容'] = fanyings
data.to_csv("../data/信访事项（1）.csv", encoding="GBK")