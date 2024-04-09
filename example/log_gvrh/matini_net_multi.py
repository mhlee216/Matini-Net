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
parser.add_argument('--gnn', type=str, default='cgcnn')
parser.add_argument('--att', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

case = f'multi_{args.gnn}'


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

data = data_list[0]

folds = KFold(n_splits=5, shuffle=True, random_state=18012019)
for k, (train_idx , valid_idx) in enumerate(folds.split(data_list)):
    train_data = [x for i, x in enumerate(data_list) if i in train_idx]
    valid_data = [x for i, x in enumerate(data_list) if i in valid_idx]
    break

def bulid_model(params):
    graph_nn = GraphNN(data, 
                       model_name=params['gnn_name'], 
                       device=args.device, 
                       dim1=params['gnn_dim'], 
                       dim2=params['gnn_dim'], 
                       dim3=params['gnn_dim'], 
                       output_dim=params['cat_dim'], 
                       gc_count=params['gc_count'])
    feature_nn = FeatureNN(data.feat.shape[1], 
                           n_fc=params['fnn_n_fc'], 
                           dim=params['fnn_dim'], 
                           output_dim=params['cat_dim'])
    pred_nn = FeatureNN(params['cat_dim']*2, 
                        n_fc=params['pred_n_fc'], 
                        dim=params['pred_dim'], 
                        output_dim=1)
    matini_model = Matini_Net(graph_nn, 
                              feature_nn, 
                              pred_nn, 
                              attention_type=params['attention_type'], 
                              n_fc=params['cat_n_fc'], 
                              dim=params['cat_dim'], 
                              n_attention=params['n_attention'], 
                              num_heads=params['num_heads'])
    return matini_model

search_space = {'gnn_name': [args.gnn], # ['cgcnn', 'gcn', 'megnet', 'mpnn', 'schnet']
                'lr': [0.001, 0.0001], 
                'weight_decay': [0.00001], 
                'gamma': [0.9], 
                'gnn_dim': [64, 128], 
                'gc_count': [2], 
                'fnn_n_fc': [4], 
                'cat_n_fc': [4], 
                'pred_n_fc': [4], 
                'fnn_dim': [128], 
                'cat_dim': [16], 
                'pred_dim': [128], 
                'n_attention': [0], # 2
                'num_heads': [0]} # 2
n_trials = mipc.get_n_trials(search_space)
print('Number of trials:', n_trials)

def objective(trial):
    params = {'device': trial.suggest_categorical('device', [args.device]), 
              'batch': trial.suggest_categorical('batch', [64]), 
              'epochs': trial.suggest_categorical('epochs', [500]), 
              'patience': trial.suggest_categorical('patience', [10]), 
              'step_size': trial.suggest_categorical('step_size', [10]), 
              'test_size': trial.suggest_categorical('test_size', [0.2]), 
              'seed': trial.suggest_categorical('seed', [18012019]), 
              'n_splits': trial.suggest_categorical('n_splits', [5]), 
              #####
              'gnn_name': trial.suggest_categorical("gnn_name", search_space['gnn_name']), 
              'lr': trial.suggest_categorical('lr', search_space['lr']), 
              'weight_decay': trial.suggest_categorical('weight_decay', search_space['weight_decay']), 
              'gamma': trial.suggest_categorical('gamma', search_space['gamma']), 
              'gnn_dim': trial.suggest_categorical('gnn_dim', search_space['gnn_dim']), 
              'gc_count': trial.suggest_categorical('gc_count', search_space['gc_count']), 
              'fnn_n_fc': trial.suggest_categorical('fnn_n_fc', search_space['fnn_n_fc']), 
              'pred_n_fc': trial.suggest_categorical('pred_n_fc', search_space['pred_n_fc']), 
              'cat_n_fc': trial.suggest_categorical('cat_n_fc', search_space['cat_n_fc']), 
              'fnn_dim': trial.suggest_categorical('fnn_dim', search_space['fnn_dim']), 
              'pred_dim': trial.suggest_categorical('pred_dim', search_space['pred_dim']), 
              'cat_dim': trial.suggest_categorical('cat_dim', search_space['cat_dim']), 
              'attention_type': trial.suggest_categorical("attention_type", [args.att]), 
              'n_attention': trial.suggest_categorical("n_attention", [0]), 
              'num_heads': trial.suggest_categorical("num_heads", [0])}
    matini_model = bulid_model(params)
    mae = mipc.optuna_train_test(params, matini_model, train_data, valid_data, prints=True, save_model=False)
    return mae

# Optuna
print('Optuna')
# optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=n_trials)
params = mipc.save_param(f'./param/param_{case}.json', study)

# Train & Test
print('Train & Valid')
matini_model = bulid_model(params)
matini_model, train_loss, valid_loss, true, pred = mipc.optuna_cross_validation(params, 
                                                                                matini_model, 
                                                                                data_list, 
                                                                                prints=True, 
                                                                                optuna=False, 
                                                                                save_model=True, 
                                                                                path=f'./model/model_{case}')

for k in range(5):
    pcc, r2, mae, mse = mipc.regre_scores(true[k], pred[k])
    valid_result = mipc.regre_save_json(f'./result/valid_{case}_{k+1}.json', pcc, r2, mae, mse, 
                                        true[k], pred[k], train_loss[k], valid_loss[k])
