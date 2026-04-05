#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:15
# @Author : ZM7
# @File : generate_neg
# @Software: PyCharm
import pandas as pd
import pickle
from utils import myFloder, pickle_loader, collate, trans_to_cuda, eval_metric, collate_test, user_neg


dataset = 'Games'
data = pd.read_csv('./Data/' + dataset + '.csv')
user = data['user_id'].unique()  #提取唯一的用户 ID。
item = data['item_id'].unique()  #提取唯一的物品 ID。
user_num = len(user) #计算唯一用户的数量。
item_num = len(item) #计算唯一物品的数量。

data_neg = user_neg(data, item_num)  #使用 user_neg 函数生成负样本
f = open(dataset+'_neg', 'wb')
pickle.dump(data_neg,f)
f.close()

