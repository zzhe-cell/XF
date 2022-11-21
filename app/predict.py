from datetime import datetime
import numpy as np
import pandas as pd
#paramList = {'start_time_list', 'end_time_list', 'weights', 'types', 'time_column', 'name_column', 'num_column', 'purpose_column'}
def get_time_window(start_time, end_time, data):
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    data[time_column] = data[time_column].astype('datetime64')
    window_data = data.loc[(data[time_column] >= start_time) & (data[time_column] <= end_time)]
    window_data[time_column] = window_data[time_column].astype(str)
    # window_data = data.loc[(data['来访时间'] >= start_time) & (data['来访时间'] <= end_time)]
    return window_data

def get_events(data, time_list, weights, types):
    if max(weights) >10 or min(weights)<1:
        raise Exception('weights out of range error')
    type_nums = []
    res_data = pd.DataFrame()
    for type in types:
        sum = 0
        for time_ in time_list:
            start_time = time_[0]
            end_time = time_[1]
            window_data = get_time_window(start_time, end_time, data)
            data1 = window_data.loc[data[purpose_column] == type]
            res_data = pd.concat([res_data, data1])
            sum += data1[num_column].sum()
        type_nums.append(sum/len(time_list))
    type_nums_weight = np.multiply(np.array(type_nums), np.array(weights)).tolist()
    type_nums_sort = np.argsort(type_nums_weight)
    type_dic_original = {}
    type_dic_weight = {}
    for i in type_nums_sort:
        type_dic_original[types[i]] = float(type_nums[i])
        type_dic_weight[types[i]] = float(type_nums_weight[i])
    type_num_original_list = []
    type_num_weight_list = []
    for i, r in res_data.iterrows():
        type_num_original_list.append(type_dic_original[r[purpose_column]])
        type_num_weight_list.append(type_dic_weight[r[purpose_column]])
    # res_data = res_data.loc[res_data['信访目的'].isin(type_dic.keys())]
    res_data['信访目的原始分数'] = type_num_original_list
    res_data['信访目的加权分数'] = type_num_weight_list
    names = res_data[name_column].unique()
    name_risks = []
    for name in names:
        data_of_name = res_data.loc[res_data[name_column] == name]
        avg_num_of_name = data_of_name.shape[0]/len(time_list)
        if avg_num_of_name >= 1:
            name_risks.append(3)
        elif avg_num_of_name >= 0.5 and avg_num_of_name<=1:
            name_risks.append(2)
        else:
            name_risks.append(1)
    name_dic = {}
    name_risks_sort = list(np.argsort(name_risks))
    for i in name_risks_sort:
        name_dic[names[i]] = int(name_risks[i])
    risk_list = []
    risk_map = {1:'低风险', 2:'中风险', 3:'高风险'}
    for index, row in res_data.iterrows():
        risk_list.append(name_dic[row[name_column]])
    res_data['风险等级'] = risk_list
    res_data = res_data.sort_values(['风险等级'], ascending=False)
    res_data['风险等级'] = res_data['风险等级'].map(risk_map)
    return res_data

def get_all_types(data):
    types = list(data['信访目的'].unique())
    # weights = np.ones(len(types)).tolist()
    # print(weights)
    dic = {"types": types}
    return dic
df[num_column] = df[num_column].astype('int')
all_types = get_all_types(df)["types"]
all_weights = np.ones(len(all_types)).tolist()
for i, type in enumerate(all_types):
    if type in types:
        all_weights[i] = weights[types.index(type)]
time_list = []
for item in zip(start_time_list, end_time_list):
    zipped = list(item)
    time_list.append(zipped)
df = get_events(df, time_list, all_weights, all_types)