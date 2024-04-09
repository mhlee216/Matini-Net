#!/usr/bin/env python
# coding: utf-8
# feature_nn.py

import torch
from torch import nn
import torch.nn.functional as F




# Multi-layer perceptron
class FeatureNN(nn.Module):
    def __init__(
        self, 
        indim, 
        direct=False, 
        data_feat=False, 
        n_fc=2, 
        dim=128, 
        output_dim=1, 
        batch_norm=True, 
        dropout_rate=0.2, 
#         attention=False, 
#         n_attention=2, 
#         num_heads=4
    ):
        super(FeatureNN, self).__init__()
        
        self.direct = direct
        self.data_feat = data_feat
#         self.attention = attention
        if direct == False:
            if n_fc < 4:
                raise ValueError("'n_fc' > 3 must be satisfied.")
        
        layer1_list = []
        layer1_list.append(nn.Linear(indim, dim))
        if batch_norm:
                layer1_list.append(nn.BatchNorm1d(dim))
        layer1_list.append(nn.ReLU())
        layer1_list.append(nn.Dropout(dropout_rate))

        for i in range(n_fc-4):
            layer1_list.append(nn.Linear(dim, dim))
            if batch_norm:
                layer1_list.append(nn.BatchNorm1d(dim))
            layer1_list.append(nn.ReLU())
            layer1_list.append(nn.Dropout(dropout_rate))
        
        self.layers1 = nn.Sequential(*layer1_list)
        
#         if self.attention:
#             fmha_list = []
#             for i in range(n_attention):
#                 fmha_list.append(nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate))
#             self.fmhas = nn.Sequential(*fmha_list)
        
        layer2_list = []
        layer2_list.append(nn.Linear(dim, int(dim/2)))
        if batch_norm:
                layer2_list.append(nn.BatchNorm1d(int(dim/2)))
        layer2_list.append(nn.ReLU())
        layer2_list.append(nn.Dropout(dropout_rate))
        layer2_list.append(nn.Linear(int(dim/2), int(dim/4)))
        layer2_list.append(nn.ReLU())
        layer2_list.append(nn.Dropout(dropout_rate))
        layer2_list.append(nn.Linear(int(dim/4), output_dim))
        self.layers2 = nn.Sequential(*layer2_list)
    
    def forward(self, data):
        
        if self.data_feat:
            feat = data.feat
        else:
            feat = data
        
        if self.direct:
            return feat
        
        else:
            x = feat

            for layer in self.layers1:
                x = layer(x)

#             if self.attention:
#                 for mha in self.fmhas:
#                     x = mha(x, x, x)[0]

            for layer in self.layers2:
                x = layer(x)

            return x




# Sine matrix with neural network
class SM(torch.nn.Module):
    def __init__(self, data, dim1=64, fc_count=1,  **kwargs):
        super(SM, self).__init__()
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_SM.shape[1], dim1)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin2 = torch.nn.Linear(dim1, 1)

    def forward(self, data):

        out = F.relu(self.lin1(data.extra_features_SM))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out




# Smooth Overlap of Atomic Positions with neural network
class SOAP(torch.nn.Module):
    def __init__(self, data, dim1, fc_count,  **kwargs):
        super(SOAP, self).__init__()
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_SOAP.shape[1], dim1)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin2 = torch.nn.Linear(dim1, 1)

    def forward(self, data):

        out = F.relu(self.lin1(data.extra_features_SOAP))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
