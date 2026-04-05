#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 3:29
# @Author : ZM7
# @File : DGSR
# @Software: PyCharm

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#############################################
# 修改：CAPE 时间编码模块替换原有的RoPE编码
#############################################
class CAPETimeEncoding(nn.Module):
    def __init__(self, hidden_size, base=10000, learnable=False):
        """
        hidden_size: 编码向量的维度
        base: 控制频率的基数，默认10000
        learnable: 是否使用可学习的频率参数
        """
        super(CAPETimeEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.base = base
        self.learnable = learnable
        
        # 初始化频率参数
        freqs = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        
        if learnable:
            # 可学习的频率参数
            self.freqs = nn.Parameter(freqs)
        else:
            # 固定的频率参数
            self.register_buffer("freqs", freqs)
    
    def forward(self, time_index):
        """
        time_index: [batch_size, seq_len]，表示时间步索引
        输出: [batch_size, seq_len, hidden_size] 的 CAPE 编码向量
        """
        # 将时间索引转换为 float 并扩展维度
        time_index = time_index.unsqueeze(-1).float()  # [batch, seq_len, 1]
        
        # 计算角度，使用CAPE的特殊频率分布
        theta = time_index * self.freqs  # [batch, seq_len, hidden_size/2]
        
        # 计算复数表示
        cos_theta = torch.cos(theta)  # [batch, seq_len, hidden_size/2]
        sin_theta = torch.sin(theta)  # [batch, seq_len, hidden_size/2]
        
        # 创建CAPE编码向量
        pos_emb = torch.zeros(time_index.shape[0], time_index.shape[1], self.hidden_size, 
                              device=time_index.device)
        pos_emb[:, :, 0::2] = cos_theta
        pos_emb[:, :, 1::2] = sin_theta
        
        return pos_emb
    
    def apply_rotary_emb(self, x, pos_emb):
        """
        将CAPE编码应用到输入向量上（使用复数旋转）
        x: 输入向量 [batch, seq_len, hidden_size]
        pos_emb: 位置编码 [batch, seq_len, hidden_size]
        """
        # 将输入向量和位置编码分为实部和虚部
        x_even = x[:, :, 0::2]  # [batch, seq_len, hidden_size/2]
        x_odd = x[:, :, 1::2]   # [batch, seq_len, hidden_size/2]
        
        cos_pos = pos_emb[:, :, 0::2]  # [batch, seq_len, hidden_size/2]
        sin_pos = pos_emb[:, :, 1::2]  # [batch, seq_len, hidden_size/2]
        
        # 应用复数旋转: (x_even + i*x_odd) * (cos + i*sin)
        # = (x_even*cos - x_odd*sin) + i*(x_even*sin + x_odd*cos)
        x_rotated_even = x_even * cos_pos - x_odd * sin_pos
        x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
        
        # 合并结果
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_rotated_even
        x_rotated[:, :, 1::2] = x_rotated_odd
        
        return x_rotated

#############################################
# 以下为原模型代码（仅修改时间编码部分）
#############################################

# DGSR 类： 负责模型的整体框架，包括：
# 用户和物品的嵌入层。
# 多层 DGSRLayers 的堆叠。
# 最终的预测层。
# forward 函数定义了数据如何在模型中流动

class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True):
        super(DGSR, self).__init__()
        self.user_num = user_num  # 用户数量
        self.item_num = item_num  # 物品数量
        self.hidden_size = input_dim  # 嵌入维度的大小
        self.item_max_length = item_max_length  # 用户和物品的最大交互长度
        self.user_max_length = user_max_length  # 用户和物品的最大交互长度
        self.layer_num = layer_num
        self.time = time  # 是否考虑时间特性
        self.last_item = last_item  # 是否使用用户的最后一个交互项作为额外特征
        # 分别表示用户和物品的长期和短期兴趣编码方式
        # long- and short-term encoder
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # update function 用户和物品的更新函数
        self.user_update = user_update
        self.item_update = item_update
        # 用于为用户和物品分别初始化嵌入矩阵，维度为 [user_num, input_dim] 和 [item_num, input_dim]
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        # unified_map：
        # 一个线性变换层，用于将不同层的用户或物品嵌入进行统一映射。
        # 如果 last_item=True，映射输入大小为 (layer_num + 1) * hidden_size（包含多层输出和最后一个交互嵌入）；否则为 layer_num * hidden_size
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        # 包含多个 DGSRLayers 子层（数量为 layer_num）。每一层负责对用户和物品的特征进行编码，捕捉长期和短期兴趣
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, 
                                                feat_drop, attn_drop, self.user_long, self.user_short, 
                                                self.item_long, self.item_short, self.user_update, self.item_update) 
                                     for _ in range(self.layer_num)])  # 多个DGSR层
        self.reset_parameters()  # 对模型参数进行初始化

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        # 对用户和物品节点初始化其特征嵌入。用户的初始嵌入存储在图中 g.nodes['user'].data['user_h']，
        # 物品的嵌入存储在 g.nodes['item'].data['item_h']
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)
                # 将每一层的用户嵌入（以及最后交互的物品嵌入）拼接起来，通过 unified_map 映射到统一维度。
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        # 通过点积计算用户嵌入和所有物品嵌入的相似度，得到推荐分数
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

# DGSRLayers 类： 负责单层图卷积或消息传递的实现，包括：
# 消息传递函数（user_message_func、item_message_func）。
# 消息聚合函数（user_reduce_func、item_reduce_func）。
# 长期和短期兴趣编码的实现。
# 特征更新函数（user_update_function、item_update_function）

class DGSRLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, 
                 user_long='orgat', user_short='att', item_long='orgat', item_short='att', 
                 user_update='residual', item_update='residual', K=4):
        super(DGSRLayers, self).__init__()
        self.hidden_size = in_feats  # 隐藏层的维度
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # user_update_m, item_update_m: 分别存储用户和物品的特征更新方式
        self.user_update_m = user_update
        self.item_update_m = item_update
        # user_max_length, item_max_length: 用户和物品序列的最大长度
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K).cuda()
        if self.user_long in ['orgat', 'gcn', 'gru'] and self.user_short in ['last', 'att', 'att1']:
            # 初始化线性层用于聚合长期和短期兴趣
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        # attention+ attention mechanism
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # -----------------------------
        # 修改部分：将原来的时间编码替换为 RoPE 时间编码
        # -----------------------------
        if self.item_long in ['orgat']:
            self.i_time_encoding = CAPETimeEncoding(self.hidden_size)
            self.i_time_encoding_k = CAPETimeEncoding(self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = CAPETimeEncoding(self.hidden_size)
            self.u_time_encoding_k = CAPETimeEncoding(self.hidden_size)

    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            # 将新旧特征拼接后通过线性层
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        # 调用 graph_update 函数更新图中所有节点的特征
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):
        # user_encoder 对 user 进行编码
        # update all nodes
        # 使用 multi_update_all 方法更新图中所有节点的特征。具体来说，它调用了 user_message_func 和 user_reduce_func 来更新用户节点的特征，
        # 调用 item_message_func 和 item_reduce_func 来更新物品节点的特征。
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    # edges.src['user_h'] 获取源节点（用户节点）的特征。
    # edges.dst['item_h'] 获取目标节点（物品节点）的特征。
    # edges.data['time'] 获取边上的时间戳信息。
    # 这些信息被打包成一个字典 dic 并返回，用于后续的聚合操作
    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    # 定义如何将邻居节点的特征聚集到目标节点。具体来说，它根据时间戳对邻居节点进行排序，并根据长期和短期兴趣建模方式（item_long, item_short, user_long, user_short）
    # 来计算目标节点的新特征。
    def item_reduce_func(self, nodes):
        h = []
        # nodes.mailbox['time']：每个消息的时间戳，形状为 (batch_size, num_neighbors)。
        # nodes.mailbox['item_h']：每个邻居的物品特征，形状为 (batch_size, num_neighbors, hidden_size)。
        # nodes.mailbox['user_h']：用户特征，形状为 (batch_size, num_neighbors, hidden_size)。
        # 先根据 time 排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['item_h'].shape[0]
        # 长期兴趣编码
        # 使用注意力机制来获取长期兴趣向量。
	
        if self.item_long == 'orgat':
            # 获取时间编码
            time_encoding = self.i_time_encoding(re_order)
            
            # 使用CAPE的旋转方式处理用户嵌入
            user_emb_rotated = self.i_time_encoding.apply_rotary_emb(nodes.mailbox['user_h'], time_encoding)
            
            # 计算注意力分数
            e_ij = torch.sum(user_emb_rotated * nodes.mailbox['item_h'], dim=2) \
                / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
                
            # 应用时间编码到输出
            time_encoding_k = self.i_time_encoding_k(re_order)
            output_rotated = self.i_time_encoding_k.apply_rotary_emb(nodes.mailbox['user_h'], time_encoding_k)
            
            h_long = torch.sum(alpha * output_rotated, dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_i(torch.cat(h, -1))}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        # 先根据 time 排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['user_h'].shape[0]
        # 长期兴趣编码
        if self.user_long == 'orgat':
                # 获取时间编码
            time_encoding = self.i_time_encoding(re_order)
            
            # 使用CAPE的旋转方式处理用户嵌入
            item_emb_rotated = self.i_time_encoding.apply_rotary_emb(nodes.mailbox['item_h'], time_encoding)
            e_ij = torch.sum(item_emb_rotated* nodes.mailbox['item_h'],
                             dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
                # 应用时间编码到输出
            time_encoding_k = self.i_time_encoding_k(re_order)
            output_rotated = self.i_time_encoding_k.apply_rotary_emb(nodes.mailbox['item_h'], time_encoding_k)
            h_long = torch.sum(alpha * output_rotated, dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h, -1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic

# 将输入数据组织为图格式，适合 DGL（Deep Graph Library）框架处理
def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long()

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), dtype=int)
    for i, u in enumerate(user):
        u = u.item()  # 将 PyTorch 张量转换为标量
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg

def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long()
