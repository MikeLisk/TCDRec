#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7 (改进优化版)
# @File : new_data_dynamic_selection.py
# @Software: PyCharm

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed

#########################################
# 辅助函数：对时间戳、次序等预处理
#########################################

# 计算 item 序列的相对次序
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

# 计算 user 序列的相对次序
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

# 对时间戳进行排序并确保时间戳的顺序不会重复或倒退
def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return data

# 根据用户和物品的交互数据生成异构图，
# 图的边数据中存储了时间戳（'time'），节点保存了 user_id 和 item_id 信息
def generate_graph(data):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    graph_data = {('item', 'by', 'user'): (torch.tensor(item), torch.tensor(user)),
                  ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data)
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))
    return graph

#####################################################
# 动态自适应邻居选择函数
#####################################################
def dynamic_select_neighbors(g, etype, nodes_dict, max_neighbors, gamma=1.0, beta=0.0):
    """
    对图 g 中指定边类型 etype 进行自适应邻居筛选：
      - 对于 nodes_dict 指定的节点集合（例如 {'user': tensor([...])}），
        遍历其入边，计算边权（此处用 'time' 属性，需转换为 float 进行计算）的均值，
        进而计算自适应门限 T = gamma * mean_weight + beta，
      - 保留权重大于门限的边；
      - 若候选边数超过 max_neighbors，则按权重降序仅保留前 max_neighbors 条边；
      - 返回筛选后的子图。
    """
    selected_eids_list = []
    node_type = list(nodes_dict.keys())[0]
    nodes = nodes_dict[node_type]
    for node in nodes.tolist():
        # 使用 in_edges 获取该节点入边信息
        src, dst, eids = g.in_edges(node, etype=etype, form='all')
        if eids.numel() == 0:
            continue
        # 取得该节点的边权，这里用 'time' 作为权重（转换为 float）
        weights = g.edges[etype].data['time'][eids].float()
        avg_weight = weights.mean().item()
        # 计算自适应门限 T
        threshold = gamma * avg_weight + beta
        # 筛选出权重大于门限的边
        mask = weights > threshold
        filtered_eids = eids[mask]
        # 若候选边数超过 max_neighbors，则取 top-k
        if filtered_eids.numel() > max_neighbors:
            filtered_weights = weights[mask]
            topk_idx = torch.argsort(filtered_weights, descending=True)[:max_neighbors]
            filtered_eids = filtered_eids[topk_idx]
        if filtered_eids.numel() > 0:
            selected_eids_list.append(filtered_eids)
    if len(selected_eids_list) > 0:
        all_selected_eids = torch.cat(selected_eids_list)
    else:
        all_selected_eids = torch.tensor([], dtype=torch.int64)
    new_subgraph = dgl.edge_subgraph(g, {etype: all_selected_eids}, relabel_nodes=False)
    return new_subgraph

#####################################################
# 根据用户生成子图数据用于训练、验证和测试
#####################################################
def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None,
                  gamma=1.0, beta=0.0):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    if len(u_seq) < 3:
        return 0, 0
    else:
        for j, t in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            # 根据当前时刻确定时间窗口范围
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(graph, edges={'by': sub_u_eid, 'pby': sub_i_eid}, relabel_nodes=False)
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])
            # 使用自适应动态邻居选择机制选择物品邻居（'by' 边）
            graph_i = dynamic_select_neighbors(sub_graph, 'by', {'user': u_temp}, item_max_length, gamma, beta)
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            for _ in range(k_hop-1):
                # 对物品节点利用 'pby' 边选择邻居用户
                graph_u = dynamic_select_neighbors(sub_graph, 'pby', {'item': i_temp}, user_max_length, gamma, beta)
                u_candidates = torch.unique(graph_u.edges(etype='pby')[0])
                # 从候选中去除已访问的用户，并取最近的 user_max_length 个
                u_temp = np.setdiff1d(u_candidates.cpu().numpy(), his_user.cpu().numpy())[-user_max_length:]
                u_temp = torch.tensor(u_temp, dtype=torch.long)
                # 利用更新后的用户集合通过 'by' 边选择物品邻居
                graph_i = dynamic_select_neighbors(sub_graph, 'by', {'user': u_temp}, item_max_length, gamma, beta)
                his_user = torch.unique(torch.cat([u_temp, his_user]))
                i_candidates = torch.unique(graph_i.edges(etype='by')[0])
                i_temp = np.setdiff1d(i_candidates.cpu().numpy(), his_item.cpu().numpy())
                i_temp = torch.tensor(i_temp, dtype=torch.long)
                his_item = torch.unique(torch.cat([i_temp, his_item]))
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            if len(edge_u) > 0:
                all_edge_u = torch.unique(torch.cat(edge_u))
            else:
                all_edge_u = torch.tensor([], dtype=torch.int64)
            if len(edge_i) > 0:
                all_edge_i = torch.unique(torch.cat(edge_i))
            else:
                all_edge_i = torch.tensor([], dtype=torch.int64)
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by': all_edge_i, 'pby': all_edge_u})
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id'] == user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id'] == last_item)[0]
            # 保存训练、验证、测试数据
            if j < split_point - 1:
                save_graphs(train_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin',
                            fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]),
                             'u_alis': u_alis, 'last_alis': last_alis})
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin',
                            fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]),
                             'u_alis': u_alis, 'last_alis': last_alis})
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin',
                            fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]),
                             'u_alis': u_alis, 'last_alis': last_alis})
                test_num += 1
        return train_num, test_num

#####################################################
# 多进程生成每个用户的子图数据
#####################################################
def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3,
                  gamma=1.0, beta=0.0):
    users = data['user_id'].unique()
    results = Parallel(n_jobs=job)(
        delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop, val_path, gamma, beta))(u)
        for u in users)
    return results

#####################################################
# 主程序入口
#####################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Games', help='data name: Games')
    parser.add_argument('--graph', action='store_true', help='是否重新生成图')
    parser.add_argument('--item_max_length', type=int, default=50, help='物品邻居最大个数')
    parser.add_argument('--user_max_length', type=int, default=50, help='用户邻居最大个数')
    parser.add_argument('--job', type=int, default=10, help='并行任务数')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    # 动态邻居选择机制参数
    parser.add_argument('--gamma', type=float, default=0.5, help='参数 gamma，用于动态门限计算')
    parser.add_argument('--beta', type=float, default=5, help='参数 beta，用于动态门限计算')
    opt = parser.parse_args()

    data_path = './Data/' + opt.data + '.csv'
    graph_path = './Data/' + opt.data + '_graph'
    data = pd.read_csv(data_path).groupby('user_id').apply(refine_time).reset_index(drop=True)
    data['time'] = data['time'].astype('int64')
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    train_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    val_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    test_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'

    print('start:', datetime.datetime.now())
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path,
                            job=opt.job, k_hop=opt.k_hop, gamma=opt.gamma, beta=opt.beta)
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    print('end:', datetime.datetime.now())