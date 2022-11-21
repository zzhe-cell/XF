# -- coding: utf-8 --**
import json

import numpy as np

from utils import Group, zhouQi, categoryEvent, highRate
from flask import Flask, jsonify, Request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
host = '0.0.0.0'
port = 5600

app = Flask(__name__)
app.config ['JSON_SORT_KEYS'] = False
CORS(app)
api = Api(app)

class SingleEventGroup(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("threshold", type=int)
        self.parser.add_argument("start_time")
        self.parser.add_argument("end_time")

    def post(self):
        data = self.parser.parse_args()
        threshold = data.get("threshold")
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        g = Group('../result/zonghe_kmeans_result.csv')
        try:
            res = g.single_event(threshold, start_time, end_time)
        except:
            res = {'message': 'error'}
        return jsonify(res)


api.add_resource(SingleEventGroup, '/single_event_group')


class ClassEventGroup(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("threshold", type=int)
        self.parser.add_argument("start_time")
        self.parser.add_argument("end_time")

    def post(self):
        data = self.parser.parse_args()
        threshold = data.get("threshold")
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        g = Group('../result/zonghe_kmeans_result.csv')
        try:
            res = g.class_event(threshold, start_time, end_time)
        except:
            res = {'message': 'error'}
        return jsonify(res)


api.add_resource(ClassEventGroup, '/class_event_group')


class ZhouqiEvent(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("start_time_list", action='append')
        self.parser.add_argument("end_time_list", action='append')
        self.parser.add_argument("k", type=int)
        self.parser.add_argument("weights", action='append')
        self.parser.add_argument("types", action='append')
        self.z = zhouQi('../data/信访事项统计表明细_xx.csv')

    def post(self):
        data = self.parser.parse_args()
        start_time_list = data.get("start_time_list")
        end_time_list = data.get("end_time_list")
        weights = data.get("weights")
        weights = [int(i) for i in weights]
        types = data.get("types")
        all_types = self.z.get_all_types()["types"]
        all_weights = np.ones(len(all_types)).tolist()
        for i, type in enumerate(all_types):
            if type in types:
                all_weights[i] = weights[types.index(type)]

        time_list = []
        for item in zip(start_time_list, end_time_list):
            zipped = list(item)
            time_list.append(zipped)
        k = data.get("k")
        try:
            res = self.z.get_events(time_list=time_list, k=k, weights=all_weights)
        except:
            res = {'message' : 'error'}
        return jsonify(res)

api.add_resource(ZhouqiEvent, '/zhou_qi')

class ZhouqiEventDetail(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("year_N", type=int)
        self.parser.add_argument("month", type=int)
        self.parser.add_argument("type", type=str)
        self.z = zhouQi('../data/信访事项统计表明细_xx.csv')
    def post(self):
        data = self.parser.parse_args()
        year_N = data.get("year_N")
        month = data.get("month")
        type = data.get("type")
        try:
            res = self.z.get_event_detail(type=type, N=year_N, m=month)
        except:
            res = {'message' : 'error'}
        return jsonify(res)


# api.add_resource(ZhouqiEventDetail, '/zhou_qi_detail')

class allTypes(Resource):

    def post(self):
        z = zhouQi('../data/信访事项统计表明细_xx.csv')
        try:
            res = z.get_all_types()
        except:
            res = {'message': 'error'}
        return jsonify(res)
api.add_resource(allTypes, '/all_types')


class jiaQuanEvent(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("start_time")
        self.parser.add_argument("end_time")
        self.parser.add_argument("k", type=int)
        self.parser.add_argument("type")
        self.parser.add_argument("year_N", type=int)
        self.z = zhouQi('../data/信访事项统计表明细_xx.csv')

    def get(self):
        data = self.parser.parse_args()
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        year_N = data.get("year_N")
        k = data.get("k")
        try:
            res = self.z.get_n_month_events(N=year_N, k=k, start_time=start_time, end_time=end_time)
        except:
            res = {'message': 'error'}
        return jsonify(res)

    def post(self):
        data = self.parser.parse_args()
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        year_N = data.get("year_N")
        type = data.get("type")
        try:
            res = self.z.get_event_detail_bytime(type=type, N=year_N, start_time=start_time, end_time=end_time)
        except:
            res = {'message': 'error'}
        return jsonify(res)
api.add_resource(jiaQuanEvent, '/jia_quan')
# 获取高频事件
# class FrequentEvent(Resource):
#     def __init__(self):
#         self.parser = reqparse.RequestParser()
#         self.parser.add_argument("threshold", type=int)
#         self.parser.add_argument("start_time")
#         self.parser.add_argument("end_time")
#
#     def post(self):
#         data = self.parser.parse_args()
#         threshold = data.get("threshold")
#         start_time = data.get("start_time")
#         end_time = data.get("end_time")
#         if threshold < 5:
#             message = '阈值过小'
#             return jsonify(message=message)
#         fe = highRate('../result/zonghe_kmeans_result.csv', start_time, end_time, threshold)
#         try:
#             res = fe.get_frequent_event()
#         except:
#            res = {'message': 'error'}
#         return jsonify(res)

class FrequentEvent(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("n", type=int)

    def post(self):
        data = self.parser.parse_args()
        n = data.get("n")
        fe = highRate('../result/zonghe_kmeans_result.csv', n)
        try:
            res = fe.get_frequent_event()
        except:
           res = {'message': 'error'}
        return jsonify(res)


api.add_resource(FrequentEvent, '/frequent_event')


# 获取某类事件
class getClassEvent(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("category")
        self.parser.add_argument("start_time")
        self.parser.add_argument("end_time")

    def post(self):
        data = self.parser.parse_args()
        category = data.get("category")
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        ce = categoryEvent('../result/zonghe_kmeans_result.csv', start_time, end_time, category)
        try:
            res = ce.get_class_event()
        except:
            res = {'message': 'error'}
        return jsonify(res)

# class getJuLeiClassEvent(Resource):
#     def __init__(self):
#         self.parser = reqparse.RequestParser()
#         self.parser.add_argument("category", type=int)
#         self.parser.add_argument("start_time")
#         self.parser.add_argument("end_time")
#
#     def post(self):
#         data = self.parser.parse_args()
#         category = data.get("category")
#         start_time = data.get("start_time")
#         end_time = data.get("end_time")
#         ce = julei_categoryEvent('../result/zonghe_kmeans_result.csv', start_time, end_time, category)
#         try:
#             res = ce.get_class_event()
#         except:
#             res = {'message': 'error'}
#         return jsonify(res)
# api.add_resource(getClassEvent, '/get_class_event')
# api.add_resource(getJuLeiClassEvent, '/get_julei_class_event')


if __name__ == '__main__':
    app.run(host = host, port = port, debug = True)
