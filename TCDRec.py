#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

# ===================== TCAN 核心组件 =====================
class TemporalCausalAugmentation(nn.Module):
    """时间因果增强网络（TCAN）模块"""
    def __init__(self, hidden_size, num_heads=4, temperature=0.1, dropout=0.1):
        super(TemporalCausalAugmentation, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.causal_mask_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.time_consistency_weight = nn.Parameter(torch.tensor(0.1))
        # 初始温度设置
        self.contrastive_temp = nn.Parameter(torch.tensor(0.1))

    def temporal_conditioned_rationale(self, history_emb, target_emb, time_emb):
        batch_size, seq_len, _ = history_emb.shape
        target_emb_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([history_emb, target_emb_expanded, time_emb], dim=-1)
        mask = self.causal_mask_generator(combined) 
        return mask

    def environment_replacement(self, rationale_emb, env_emb, strategy='spatial_temporal'):
        batch_size, seq_len, hidden_size = env_emb.shape
        
        if strategy == 'spatial':
            idx = torch.randperm(batch_size, device=env_emb.device)
            replaced_env = env_emb[idx, random.randint(0, seq_len-1), :]
            return rationale_emb + replaced_env
        elif strategy == 'temporal':
            time_idx = torch.randint(0, seq_len, (batch_size,), device=env_emb.device)
            replaced_env = env_emb[torch.arange(batch_size, device=env_emb.device), time_idx, :]
            return rationale_emb + replaced_env
        else:  
            batch_idx = torch.randperm(batch_size, device=env_emb.device)
            time_idx = torch.randint(0, seq_len, (batch_size,), device=env_emb.device)
            replaced_env = env_emb[batch_idx, time_idx, :]
            return rationale_emb + replaced_env

    def forward(self, current_emb, history_emb, time_emb):
        batch_size, seq_len, hidden_size = history_emb.shape
        
        mask = self.temporal_conditioned_rationale(history_emb, current_emb, time_emb)
        rationale_emb = (mask * history_emb).sum(dim=1) 
        env_emb = ((1 - mask) * history_emb) 
        
        # 【核心修复1】：完美处理 Train 与 Test 的分支逻辑，测试时利用因果推断期望，抛弃随机扰动
        if self.training:
            strategy = random.choice(['spatial', 'temporal', 'spatial_temporal'])
            augmented_emb = self.environment_replacement(rationale_emb, env_emb, strategy)
            
            temp = torch.clamp(self.contrastive_temp, min=0.05)
            
            positive_sim = F.cosine_similarity(augmented_emb, rationale_emb, dim=-1) / temp
            negative_sim = F.cosine_similarity(augmented_emb, env_emb.mean(dim=1), dim=-1) / temp
            
            logits = torch.stack([positive_sim, negative_sim], dim=1) 
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device) 
            contrastive_loss = F.cross_entropy(logits, labels, reduction='none')
            
            total_loss = contrastive_loss.view(-1, 1)
        else:
            # 评估模式：保留因果表征与环境平滑，让向量处于和训练相同的特征空间
            augmented_emb = rationale_emb + env_emb.mean(dim=1)
            total_loss = torch.zeros(batch_size, 1, device=history_emb.device)
            
        return augmented_emb, mask, total_loss

class LearnableCyclicBasis(nn.Module):
    def __init__(self, input_dim, num_cycles=8, max_period=168):  
        super(LearnableCyclicBasis, self).__init__()
        self.input_dim = input_dim
        self.num_cycles = num_cycles
        self.max_period = max_period
        
        self.log_frequencies = nn.Parameter(torch.log(torch.linspace(1.0, max_period, num_cycles)))
        self.phases = nn.Parameter(torch.randn(num_cycles) * 0.1)
        self.amplitudes = nn.Parameter(torch.ones(num_cycles))
        self.projection = nn.Linear(num_cycles * 2, input_dim)
        
    def forward(self, timestamps):
        original_shape = timestamps.shape
        if len(original_shape) == 1:
            timestamps = timestamps.unsqueeze(1)
        
        batch_size, seq_len = timestamps.shape
        frequencies = torch.clamp(torch.exp(self.log_frequencies), min=1e-5)
        
        t_expanded = timestamps.unsqueeze(-1)  
        freq_expanded = frequencies.unsqueeze(0).unsqueeze(0)  
        phase_expanded = self.phases.unsqueeze(0).unsqueeze(0)  
        amp_expanded = self.amplitudes.unsqueeze(0).unsqueeze(0)  
        
        angles = 2 * math.pi * t_expanded / freq_expanded + phase_expanded
        sin_vals = amp_expanded * torch.sin(angles)  
        cos_vals = amp_expanded * torch.cos(angles)  
        
        cyclic_features = torch.cat([sin_vals, cos_vals], dim=-1)  
        cyclic_embed = self.projection(cyclic_features)  
        
        if len(original_shape) == 1:
            cyclic_embed = cyclic_embed.squeeze(1)
        return cyclic_embed

class CyclicTime2Vec(nn.Module):
    def __init__(self, input_dim, max_length):
        super(CyclicTime2Vec, self).__init__()
        self.embed_dim = input_dim
        self.max_length = max_length
        linear_dim = input_dim // 4
        periodic_dim = input_dim // 4
        cyclic_dim = input_dim - linear_dim - periodic_dim  
        
        self.linear = nn.Linear(1, linear_dim)
        self.periodic = nn.Linear(periodic_dim, periodic_dim) 
        self.freq = nn.Parameter(torch.randn(periodic_dim) * 0.1)
        self.phase = nn.Parameter(torch.randn(periodic_dim) * 0.1)
        
        self.cyclic_basis = LearnableCyclicBasis(cyclic_dim, num_cycles=6)
        
    def forward(self, tau):
        original_shape = tau.shape
        tau_flat = tau.view(-1, 1)  
        
        linear_part = self.linear(tau_flat)  
        periodic_part = torch.sin(tau_flat * self.freq.unsqueeze(0) + self.phase.unsqueeze(0))  
        periodic_part = self.periodic(periodic_part)
        cyclic_part = self.cyclic_basis(tau.view(-1))  
        
        time_embed = torch.cat([linear_part, periodic_part, cyclic_part], dim=-1)
        target_shape = original_shape + (self.embed_dim,)
        time_embed = time_embed.view(target_shape)
        
        return time_embed

class DualGatingMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(DualGatingMechanism, self).__init__()
        self.hidden_size = hidden_size
        
        self.struct_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.time_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, struct_repr, temp_repr, cyclic_embed):
        struct_input = torch.cat([struct_repr, cyclic_embed], dim=-1)
        struct_gate_weight = self.struct_gate(struct_input)
        gated_struct = struct_gate_weight * struct_repr
        
        temp_input = torch.cat([temp_repr, cyclic_embed], dim=-1)
        temp_gate_weight = self.time_gate(temp_input)
        gated_temp = temp_gate_weight * temp_repr
        
        fusion_input = torch.cat([gated_struct, gated_temp, cyclic_embed], dim=-1)
        fusion_weight = self.fusion_gate(fusion_input)
        
        final_repr = fusion_weight * gated_struct + (1 - fusion_weight) * gated_temp
        return final_repr, fusion_weight


class DGSTGN(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, 
                 feat_drop=0.2, attn_drop=0.2, user_long='dg_stgn', item_long='dg_stgn',
                 user_update='rnn', item_update='rnn', last_item=True, layer_num=3, 
                 time=True, num_heads=5, tcan_enabled=True):
        super(DGSTGN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        self.tcan_enabled = tcan_enabled
        
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
            
        # 【核心修复3】：删除了破坏内积度量的单边 LayerNorm
        
        self.layers = nn.ModuleList([
            DGSTGNLayers(self.hidden_size, self.hidden_size, self.user_max_length,
                        self.item_max_length, feat_drop, attn_drop, user_long,
                        item_long, user_update, item_update, num_heads=num_heads, tcan_enabled=tcan_enabled)
            for _ in range(self.layer_num)
        ])
        
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        device = self.user_embedding.weight.device
        
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].to(device))
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].to(device))
        
        batch_tcan_loss = torch.tensor(0.0, device=device)
        
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                
                if self.tcan_enabled and is_training:
                    u_loss = g.nodes['user'].data.get('tcan_loss', torch.zeros(1, device=device)).mean()
                    i_loss = g.nodes['item'].data.get('tcan_loss', torch.zeros(1, device=device)).mean()
                    batch_tcan_loss = batch_tcan_loss + u_loss + i_loss
                    
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)
        
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        
        if is_training:
            return score, batch_tcan_loss
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

class DGSTGNLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, 
                 feat_drop=0.2, attn_drop=0.2, user_long='dg_stgn', item_long='dg_stgn',
                 user_update='residual', item_update='residual', K=4, num_heads=5, tcan_enabled=True):
        super(DGSTGNLayers, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_feats // num_heads
        self.hidden_size = in_feats
        self.user_long = user_long
        self.item_long = item_long
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.tcan_enabled = tcan_enabled
        
        self.time2vec = CyclicTime2Vec(in_feats, max(user_max_length, item_max_length))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        if self.user_long in ['dg_stgn']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['dg_stgn']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        
        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
            
        self.combine_gate_u = nn.Linear(2 * in_feats, in_feats)
        self.combine_gate_i = nn.Linear(2 * in_feats, in_feats) 

        # User Modules
        if self.user_long in ['dg_stgn']:
            self.user_q_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            self.user_k_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            self.user_v_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            
            self.struct_attn_u = nn.ModuleList([nn.Linear(in_feats, self.head_dim, bias=False) for _ in range(num_heads)])
            self.temp_proj_u = nn.Linear(num_heads * self.head_dim, in_feats)
            self.struct_proj_u = nn.Linear(num_heads * self.head_dim, in_feats)
            self.dual_gate_u = DualGatingMechanism(in_feats)
        
        # Item Modules
        if self.item_long in ['dg_stgn']:
            self.item_q_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            self.item_k_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            self.item_v_proj = nn.Linear(in_feats, num_heads * self.head_dim, bias=False)
            
            self.struct_attn_i = nn.ModuleList([nn.Linear(in_feats, self.head_dim, bias=False) for _ in range(num_heads)])
            self.temp_proj_i = nn.Linear(num_heads * self.head_dim, in_feats)
            self.struct_proj_i = nn.Linear(num_heads * self.head_dim, in_feats)
            self.dual_gate_i = DualGatingMechanism(in_feats)
            
        if self.tcan_enabled:
            self.tcan_user = TemporalCausalAugmentation(in_feats)
            self.tcan_item = TemporalCausalAugmentation(in_feats)

    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual': return F.elu(user_now + user_old)
        elif self.user_update_m == 'concat': return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'norm': return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn': return torch.tanh(self.user_update(torch.cat([user_now, user_old], -1)))

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual': return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat': return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'norm': return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn': return torch.tanh(self.item_update(torch.cat([item_now, item_old], -1)))

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user']
            item_ = feat_dict['item']
        
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        
        g = self.graph_update(g)
        
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        
        return {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}

    def graph_update(self, g):
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        return {'time': edges.data['time'], 'user_h': edges.src['user_h'], 'item_h': edges.dst['item_h']}

    def item_reduce_func(self, nodes):
        h = []
        batch_size = nodes.mailbox['user_h'].shape[0]
        seq_len = nodes.mailbox['user_h'].shape[1]
        
        if self.item_long == 'dg_stgn':
            # 【核心修复2】：复原了原始的时间表达，不再使用会剥夺绝对间隔信息的 Min-Max 归一化
            time_embed = self.time2vec(nodes.mailbox['time'].float()) 
            
            user_h = nodes.mailbox['user_h']
            
            q = self.item_q_proj(nodes.data['item_h']).view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.item_k_proj(user_h).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.item_v_proj(user_h).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            time_embed_permuted = time_embed.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            k_time = k + time_embed_permuted
            attn_scores = torch.matmul(q, k_time.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = self.atten_drop(F.softmax(attn_scores, dim=-1))
            context = torch.matmul(attn_weights, v)
            h_temp = context.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_heads * self.head_dim)
            h_temp = self.temp_proj_i(h_temp) 
            
            struct_outs = []
            for head in range(self.num_heads):
                q_struct = self.struct_attn_i[head](nodes.data['item_h']) 
                k_struct = self.struct_attn_i[head](user_h)               
                attn_scores_struct = torch.matmul(q_struct.unsqueeze(1), k_struct.transpose(1, 2))
                alpha_struct = F.softmax(attn_scores_struct / math.sqrt(self.head_dim), dim=-1)
                struct_out = torch.matmul(alpha_struct, k_struct).squeeze(1)
                struct_outs.append(struct_out)
            
            h_struct = torch.cat(struct_outs, dim=1)
            h_struct = self.struct_proj_i(h_struct) 
            
            cyclic_embed = time_embed.mean(dim=1)
            h_combined, gate_weight = self.dual_gate_i(h_struct, h_temp, cyclic_embed) 
            
            tcan_loss_val = torch.zeros(batch_size, 1, device=nodes.data['item_h'].device)
            # 无差别应用增强推断，抹除 eval() 的条件屏障
            if self.tcan_enabled:
                augmented_emb, _, tcan_loss_val = self.tcan_item(
                    current_emb=nodes.data['item_h'], 
                    history_emb=user_h, 
                    time_emb=time_embed
                )
                h_combined = h_combined + augmented_emb 
                
            h.append(h_combined)
        
        output_dict = {'item_h': h[0] if len(h) == 1 else self.agg_gate_i(torch.cat(h, -1))}
        if self.tcan_enabled and self.training:
            output_dict['tcan_loss'] = tcan_loss_val
        return output_dict

    def user_message_func(self, edges):
        return {'time': edges.data['time'], 'item_h': edges.src['item_h'], 'user_h': edges.dst['user_h']}

    def user_reduce_func(self, nodes):
        h = []
        batch_size = nodes.mailbox['item_h'].shape[0]
        seq_len = nodes.mailbox['item_h'].shape[1]
        
        if self.user_long == 'dg_stgn':
            time_embed = self.time2vec(nodes.mailbox['time'].float()) 
            
            item_h = nodes.mailbox['item_h']
            
            q = self.user_q_proj(nodes.data['user_h']).view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.user_k_proj(item_h).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.user_v_proj(item_h).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            time_embed_permuted = time_embed.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k_time = k + time_embed_permuted
            attn_scores = torch.matmul(q, k_time.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = self.atten_drop(F.softmax(attn_scores, dim=-1))
            context = torch.matmul(attn_weights, v)
            h_temp = context.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_heads * self.head_dim)
            h_temp = self.temp_proj_u(h_temp) 
            
            struct_outs = []
            for head in range(self.num_heads):
                q_struct = self.struct_attn_u[head](nodes.data['user_h'])
                k_struct = self.struct_attn_u[head](item_h)
                attn_scores_struct = torch.matmul(q_struct.unsqueeze(1), k_struct.transpose(1, 2))
                alpha_struct = F.softmax(attn_scores_struct / math.sqrt(self.head_dim), dim=-1)
                struct_out = torch.matmul(alpha_struct, k_struct).squeeze(1)
                struct_outs.append(struct_out)
            
            h_struct = torch.cat(struct_outs, dim=1)
            h_struct = self.struct_proj_u(h_struct)
            
            cyclic_embed = time_embed.mean(dim=1)
            h_combined, gate_weight = self.dual_gate_u(h_struct, h_temp, cyclic_embed)
            
            tcan_loss_val = torch.zeros(batch_size, 1, device=nodes.data['user_h'].device)
            # 无差别应用增强推断，抹除 eval() 的条件屏障
            if self.tcan_enabled:
                augmented_emb, _, tcan_loss_val = self.tcan_user(
                    current_emb=nodes.data['user_h'], 
                    history_emb=item_h, 
                    time_emb=time_embed
                )
                h_combined = h_combined + augmented_emb 
                
            h.append(h_combined)
        
        output_dict = {'user_h': h[0] if len(h) == 1 else self.agg_gate_u(torch.cat(h, -1))}
        if self.tcan_enabled and self.training:
            output_dict['tcan_loss'] = tcan_loss_val
        return output_dict

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

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
        u = u.item()
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg

def collate_test(data, user_neg):
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

TCDRec = DGSTGN