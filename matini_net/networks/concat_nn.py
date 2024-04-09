#!/usr/bin/env python
# coding: utf-8
# attention_nn.py

import torch
from torch import nn
import torch.nn.functional as F




# Multi-modal Multi-head Attention Neural Network
class Matini_Net(nn.Module):
    def __init__(
        self, 
        graph_nn, 
        feature_nn, 
        pred_nn, 
        attention_type=1, 
        n_fc=2, 
        dim=16, 
        output_dim=1, 
        batch_norm=True, 
        dropout_rate=0.2, 
        n_attention=2, 
        num_heads=4
    ):
        super(Matini_Net, self).__init__()
        
        self.graph_nn = graph_nn
        self.feature_nn = feature_nn
        self.pred_nn = pred_nn
                
        if attention_type not in [0, 1, 2, 3]:
            raise ValueError("'attention_type' must only be selected from: [0, 1, 2, 3]")
        self.attention_type = attention_type
        
        gmha_list = []
        for i in range(n_attention):
            gmha_list.append(nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate))
        self.gmhas = nn.Sequential(*gmha_list)
        
        fmha_list = []
        for i in range(n_attention):
            fmha_list.append(nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate))
        self.fmhas = nn.Sequential(*fmha_list)
        
        if self.attention_type in [1, 3]:
            pre_layer_list = []
            pre_layer_list.append(nn.Linear(dim*4, dim*2))
            if batch_norm:
                pre_layer_list.append(nn.BatchNorm1d(dim*2))
            pre_layer_list.append(nn.ReLU())
            pre_layer_list.append(nn.Dropout(dropout_rate))
            self.pre_layers = nn.Sequential(*pre_layer_list)
    
    def forward(self, data):
        
        g = self.graph_nn(data)
        f = self.feature_nn(data.feat)
        
        if self.attention_type in [1, 3]:
            raw_g = g
            raw_f = f
        
        if self.attention_type == 0:
            x = torch.cat((g, f), dim=1)
        
        elif self.attention_type == 1:
            for mha in self.gmhas:
                g = mha(g, f, f)[0]
            for mha in self.fmhas:
                f = mha(f, g, g)[0]
            x = torch.cat((raw_g, g, f, raw_f), dim=1)
            for layer in self.pre_layers:
                x = layer(x)
        
        elif self.attention_type == 2:
            for mha in self.gmhas:
                g = mha(g, f, f)[0]
            for mha in self.fmhas:
                f = mha(f, g, g)[0]
            x = torch.cat((g, f), dim=1)

        elif self.attention_type == 3:
            for mha in self.gmhas:
                g = mha(g, g, g)[0]
            for mha in self.fmhas:
                f = mha(f, f, f)[0]
            x = torch.cat((raw_g, g, f, raw_f), dim=1)
            for layer in self.pre_layers:
                x = layer(x)

        x = self.pred_nn(x)

        return x
    
    def attention_type_description():
        print('if attention_type == 0:')
        print('\tg = graph_nn(input1)')
        print('\tf = feature_nn(input2)')
        print('\tx = concat((g, f), dim=1)')
        print('\toutput = pred_nn(x)')
        print()
        print('if attention_type == 1:')
        print('\tg = graph_nn(input1)')
        print('\tf = feature_nn(input2)')
        print('\traw_g = g')
        print('\traw_f = f')
        print('\tattg = multi_head_attention(g, f, f)')
        print('\tattf = multi_head_attention(f, g, g)')
        print('\tx = concat((raw_g, g, f, raw_f), dim=1)')
        print('\toutput = pred_nn(x)')
        print()
        print('if attention_type == 2:')
        print('\tg = graph_nn(input1)')
        print('\tf = feature_nn(input2)')
        print('\tattg = multi_head_attention(g, f, f)')
        print('\tattf = multi_head_attention(f, g, g)')
        print('\tx = concat((g, f), dim=1)')
        print('\toutput = pred_nn(x)')
        print()
        print('if attention_type == 3:')
        print('\tg = graph_nn(input1)')
        print('\tf = feature_nn(input2)')
        print('\traw_g = g')
        print('\traw_f = f')
        print('\tattg = multi_head_attention(g, g, g)')
        print('\tattf = multi_head_attention(f, f, f)')
        print('\tx = concat((raw_g, g, f, raw_f), dim=1)')
        print('\toutput = pred_nn(x)')
