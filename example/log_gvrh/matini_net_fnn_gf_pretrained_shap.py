#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import KFold

import os
import sys
sys.path.append("/home/mhlee/MatiniNet/")
import matini_net.data as midt
import matini_net.database as midb
from matini_net.networks.graph_nn import GraphNN
from matini_net.networks.feature_nn import FeatureNN
from matini_net.networks.concat_nn import Matini_Net
import matini_net.process as mipc

import shap
import matplotlib.pyplot as plt
import optuna

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./data/')
parser.add_argument('--target', type=str, default='target')
parser.add_argument('--nan', type=str, default='dropna')
parser.add_argument('--case', type=str, default='fnn_gf_pretrained_cgcnn')
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

args.gnn = args.case.split('_')[-1]


# Data
print('Data')
files = [args.input+f for f in os.listdir(args.input) if 'csv' in f]
df = pd.concat([pd.read_csv(f) for f in files]).reset_index(drop=True)
if args.nan == 'dropna':
    df = df.dropna(axis=1)
else:
     df = df.fillna(0)  
target_col = args.target

df, feature_df = midt.split_feature_df(df, ['id', 'poscar', target_col])
data_list = midt.get_data_list(df, feature_df, target_col)

graph_nn = mipc.model_load(f'./model/model_gnn_{args.gnn}_1.pt')
graph_nn = midt.get_pretrained_graph_featurizer(graph_nn)
data_list = midt.get_graph_features(data_list, graph_nn, 64, args.device)

data = data_list[0]

folds = KFold(n_splits=5, shuffle=True, random_state=18012019)
for k, (train_idx , valid_idx) in enumerate(folds.split(data_list)):
    train_data = [x for i, x in enumerate(data_list) if i in train_idx]
    valid_data = [x for i, x in enumerate(data_list) if i in valid_idx]
    break

best_nn = mipc.model_load(f'./model/model_{args.case}_{args.k}.pt')
best_nn = best_nn.to(args.device)

# SHAP
print('SHAP')
best_nn.data_feat = False
train_x_list = mipc.shap_tensor_single(train_data, args.device)
valid_x_list = mipc.shap_tensor_single(valid_data, args.device)
shap_values = mipc.shap_explainer(best_nn, train_x_list, valid_x_list)
mipc.save_shap_data(f'./shap/shap_values_{args.case}_{args.k}.json', shap_values, valid_x_list)
# shap_values, valid_x_list = mipc.load_shap_data(f'./shap/shap_values_{args.case}.json')

# shap.summary_plot(shap_values, valid_x_list, plot_size=[20, 10], show=False, color_bar=False)
# cbar = plt.colorbar()
# cbar.set_label('Feature value', fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('SHAP value', fontsize=20)
# plt.savefig(f'./shap/shap_{args.case}.png')
# # plt.show()
