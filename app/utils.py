from datetime import datetime
import pandas as pd
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=Warning)

class Group:
    def __init__(self, filepath):
        self.filepath = filepath

    def single_event(self, threshold, start_time, end_time):
        data = pd.read_csv(self.filepath, encoding='utf-8')
        data = data.drop(columns=['反映内容', '类别', '关键词'])
        data = get_time_window(start_time, end_time, data)
        data = data[data['人数'] >= threshold]
        data = data.reset_index(drop=True)
        # new_data = data.groupby(by='人数').count()
        rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
        data.columns = rename
        dic = {}
        dic['data'] = data.to_dict(orient="records")
        return dic


    def class_event(self, threshold, start_time, end_time):
        data = pd.read_csv(self.filepath, encoding='utf-8')
        data = data.drop(columns=['反映内容', '关键词'])
        data = get_time_window(start_time, end_time, data)
        type_list = data['类别'].unique().tolist()
        groups = data.groupby(by='类别')
        dic = {}
        res_data = pd.DataFrame()
        categorys = []
        nums = []

        for type in type_list:
            data_for_type = groups.get_group(type).reset_index(drop=True)
            data_for_type_noduplicate = data_for_type.drop_duplicates(subset=['来访人'], keep='first')
            num_for_type = int(data_for_type_noduplicate['人数'].sum())
            if num_for_type >= threshold:
                # 统计最多的信访目的作为类名
                nums.append(num_for_type)
                category = data_for_type['信访目的'].mode()[0]
                data_for_type['category'] = str(category)+'_'+str(type)
                categorys.append(str(category)+'_'+str(type))
                res_data = pd.concat([res_data, data_for_type])
        res_data = res_data.drop(columns="类别")
        rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status', 'type',]
        res_data.columns = rename
        res_data = res_data.to_dict(orient="records")
        dic['data'] = res_data
        dic['nums'] = nums
        dic['categorys'] = categorys
        return dic
        # data = pd.read_csv(self.filepath, encoding='utf-8')
        # data = data.drop(columns=['反映内容', '关键词'])
        # data = data.sort_values(["来访时间"])
        # rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
        #
        # window_data = get_time_window(start_time, end_time, data)
        #
        # event_dict = {}
        # for index, row in window_data.iterrows():
        #     if row['信访目的'] not in event_dict.keys():
        #         event_dict[row['信访目的']] = 0
        #     event_dict[row['信访目的']] += row['人数']
        # temp = []
        # for key in event_dict.keys():
        #     if event_dict[key] < threshold:
        #         temp.append(key)
        # for key in temp:
        #     event_dict.pop(key)
        # event = []
        # window_data = window_data[window_data['信访目的'].isin(event_dict.keys())]
        # window_data.columns = rename
        # for index, row in window_data.iterrows():
        #     event.append(row.to_dict())
        # dic = {'data': event}
        # return dic

# class highRate:
#
#     def __init__(self, filepath, start_time, end_time, threshold):
#         self.filepath = filepath
#         self.start_time = start_time
#         self.end_time = end_time
#         self.threshold = threshold
#
#     def get_frequent_event(self):
#             data = pd.read_csv(self.filepath, encoding='utf-8')
#             data = data.drop(columns=['反映内容', '类别', '关键词'])
#             data = data.sort_values(["来访时间"])
#             rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
#
#             window_data = get_time_window(self.start_time, self.end_time, data)
#
#             event_dict = {}
#             for index, row in window_data.iterrows():
#                 if row['信访目的'] not in event_dict.keys():
#                     event_dict[row['信访目的']] = 1
#                 event_dict[row['信访目的']] += 1
#             temp = []
#             for key in event_dict.keys():
#                 if event_dict[key] < self.threshold:
#                     temp.append(key)
#             for key in temp:
#                 event_dict.pop(key)
#             event = []
#             window_data = window_data[window_data['信访目的'].isin(event_dict.keys())]
#             window_data.columns = rename
#             for index, row in window_data.iterrows():
#                 event.append(row.to_dict())
#             return event

class highRate():

    def __init__(self, filepath, n):
        self.filepath = filepath
        self.n = n

    def get_frequent_event(self):
        data = pd.read_csv(self.filepath, encoding='utf-8')
        data = data.drop(columns=['反映内容', '类别', '关键词'])
        end_year = time.localtime(time.time())[0]
        start_year = end_year
        end_month = time.localtime(time.time())[1]
        start_month = end_month
        for i in range(self.n):
            if start_month == 1:
                start_year -= 1
                month = 12
            else:
                start_month -= 1
        start_time = str(start_year) + '-' + str(start_month) + '-1'
        end_time =  str(end_year) + '-' + str(end_month) + '-1'
        data = data.sort_values(["来访时间"])
        window_data = get_time_window(start_time, end_time, data)
        rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
        window_data.columns = rename
        window_data = window_data.to_dict("records")
        dic = {"data": window_data}
        return dic






class categoryEvent:

    def __init__(self, filepath, start_time, end_time, category):
        self.filepath = filepath
        self.start_time = start_time
        self.end_time = end_time
        self.category = category

    def get_class_event(self):
        data = pd.read_csv(self.filepath, encoding='utf-8')
        data = data.drop(columns=['反映内容', '类别', '关键词'])
        data = data.sort_values(["来访时间"])
        window_data = get_time_window(self.start_time, self.end_time, data)
        category_data = window_data.loc[window_data['信访目的'] == self.category]
        rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
        category_data.columns = rename
        event_dict = {}
        event_dict['data'] = category_data.to_dict('records')
        return event_dict


#
# def get_class_event(data, start_time, end_time, category):
#     data = data.drop(columns=['反映内容', '关键词'])
#     data = data.sort_values(["来访时间"])
#     window_data = get_time_window(start_time, end_time, data)
#     category_data = window_data.loc[window_data['类别'] == category]
#     category_data = category_data.drop(columns='类别')
#     rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
#     category_data.columns = rename
#     event_dict = {}
#     event_dict['data'] = category_data.to_dict('records')
#     return event_dict

def get_time_window(start_time, end_time, data):
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    data['来访时间'] = data['来访时间'].astype('datetime64')
    window_data = data.loc[(data['来访时间'] >= start_time) & (data['来访时间'] <= end_time)]
    window_data['来访时间'] = window_data['来访时间'].astype(str)
    # window_data = data.loc[(data['来访时间'] >= start_time) & (data['来访时间'] <= end_time)]
    return window_data


class zhouQi:
    def __init__(self, filepath):
        self.filepath = filepath
    def get_events(self,  time_list, k, weights):
        if max(weights) >10 or min(weights)<1:
            raise Exception("error")
        data = pd.read_csv(self.filepath, encoding='gbk')
        types = data['信访目的'].unique()
        type_nums = []
        res_data = pd.DataFrame()
        for type in types:
            sum = 0
            for time_ in time_list:
                start_time = time_[0]
                end_time = time_[1]
                window_data = get_time_window(start_time, end_time, data)
                data1 = window_data.loc[data['信访目的'] == type]
                res_data = pd.concat([res_data, data1])
                sum += data1['人数'].sum()
            type_nums.append(sum/len(time_list))
        type_nums_weight = np.multiply(np.array(type_nums), np.array(weights)).tolist()
        type_nums_sort = np.argsort(type_nums_weight)[-k:]
        type_dic_original = {}
        type_dic_weight = {}
        for i in type_nums_sort:
            type_dic_original[types[i]] = float(type_nums[i])
            type_dic_weight[types[i]] = float(type_nums_weight[i])

        # res_data = res_data.loc[res_data['信访目的'].isin(type_dic.keys())]
        names = res_data['来访人'].unique()
        name_risks = []
        for name in names:
            data_of_name = res_data.loc[res_data['来访人'] == name]
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
            risk_list.append(name_dic[row['来访人']])
        rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
        res_data = res_data.drop(columns='序号')
        res_data.columns = rename
        res_data['risk'] = risk_list
        res_data = res_data.sort_values(["risk"], ascending=False)
        res_data['risk'] = res_data['risk'].map(risk_map)
        dic = {'name': name_dic, 'type_original': type_dic_original, 'type_weight': type_dic_weight, 'data': res_data.to_dict('records')}
        return dic

    def get_all_types(self):
        data = pd.read_csv(self.filepath, encoding='gbk')
        types = list(data['信访目的'].unique())
        # weights = np.ones(len(types)).tolist()
        # print(weights)
        dic = {"types": types}
        return dic





    # def get_event_detail(self, type, N, m):
    #     data = pd.read_csv(self.filepath, encoding='gbk')
    #     data['来访时间'] = pd.to_datetime(data['来访时间'])
    #     data['year'] = data['来访时间'].dt.year
    #     data['month'] = data['来访时间'].dt.month
    #     end_year = time.localtime(time.time())[0]
    #     start_year = end_year - N + 1
    #     data = data.loc[(data['year'] >= start_year) & (data['year'] <= end_year) & (data['month'] == m) & (data['信访目的'] == type)]
    #     data = data.drop(columns=['year', 'month', '序号'])
    #     data = data.sort_values('来访时间')
    #     data['来访时间'] = data['来访时间'].dt.strftime('%Y-%m-%d')
    #     rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
    #     data.columns = rename
    #     dic = {}
    #     dic['data'] = data.to_dict(orient="records")
    #     return dic
    #
    # def get_n_month_events(self, N, start_time, end_time, k=5):
    #     data = pd.read_csv(self.filepath, encoding='gbk')
    #
    #     data['来访时间'] = pd.to_datetime(data['来访时间'])
    #     data['year'] = data['来访时间'].dt.year
    #     end_year = time.localtime(time.time())[0]
    #     start_year = end_year - N + 1
    #     data = data.loc[(data['year'] >= start_year) & (data['year'] <= end_year)]
    #     types = data['信访目的'].unique()
    #     type_months = []
    #     for type in types:
    #         sum = 0
    #         for y in range(start_year, end_year + 1):
    #             start_time_ = datetime.strptime(str(y) + '-' + start_time, '%Y-%m-%d')
    #             end_time_ = datetime.strptime(str(y) + '-' + end_time, '%Y-%m-%d')
    #             data1 = data.loc[(data['year'] == y) & (data['来访时间'] >= start_time_) & (data['来访时间'] <= end_time_) & (data['信访目的'] == type)]
    #             sum += data1['人数'].sum()
    #         type_months.append(sum)
    #     type_month_sort = np.argsort(type_months)[:-(k + 1):-1]
    #     dic = {}
    #     for i in type_month_sort:
    #         dic[types[i]] = int(type_months[i])
    #     return dic
    #
    # def get_event_detail_bytime(self, type, N, start_time, end_time):
    #     data = pd.read_csv(self.filepath, encoding='gbk')
    #     data['来访时间'] = pd.to_datetime(data['来访时间'])
    #     data['year'] = data['来访时间'].dt.year
    #     end_year = time.localtime(time.time())[0]
    #     start_year = end_year - N + 1
    #     data1 = pd.DataFrame()
    #     for y in range(start_year, end_year + 1):
    #         start_time_ = datetime.strptime(str(y) + '-' + start_time, '%Y-%m-%d')
    #         end_time_ = datetime.strptime(str(y) + '-' + end_time, '%Y-%m-%d')
    #         data1 = data1.append(data.loc[(data['year'] == y) & (data['来访时间'] >= start_time_) & (data['来访时间'] <= end_time_) & (data['信访目的'] == type)])
    #     data1 = data1.drop(columns=['序号', 'year'])
    #     data1 = data1.sort_values('来访时间')
    #     rename = ['registration', 'code', 'people', 'number', 'purpose', 'related', 'detail', 'time', 'status']
    #     data1.columns = rename
    #     dic = {}
    #     dic['data'] = data1.to_dict(orient="records")
    #     return dic

# zhouqi = zhouQi('../data/信访事项统计表明细_xx.csv')
# _, weights = zhouqi.get_all_types()
# d = zhouqi.get_events(time_list=[['2022-1-1', '2022-2-1'], ['2022-2-2', '2022-3-1'], ['2022-3-2', '2022-4-1'], ['2022-4-2', '2022-6-1']], k=5, weights=weights )
# #d = zhouqi.get_event_detail_bytime(N=2, type='经济纠纷', start_time='1-1', end_time='6-1')
# group = Group('../result/zonghe_kmeans_result.csv')
# result = group.class_event(start_time='2022-1-1', end_time='2022-6-1', threshold=3)
# ce = julei_categoryEvent('../result/zonghe_kmeans_result.csv', '2022-1-1', "2022-10-1", 10)
# re = ce.get_class_event()
# # print(result)