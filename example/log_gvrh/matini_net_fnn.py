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
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

case = 'fnn'


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
    matini_model = FeatureNN(data.feat.shape[1], 
                             data_feat=True, 
                             n_fc=params['fnn_n_fc'], 
                             dim=params['fnn_dim'], 
                             output_dim=1)
    return matini_model

search_space = {'lr': [0.001, 0.0001], 
                'weight_decay': [0.00001], 
                'gamma': [0.9], 
                'fnn_n_fc': [4, 6], 
                'fnn_dim': [512, 1024]}
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
              'lr': trial.suggest_categorical('lr', search_space['lr']), 
              'weight_decay': trial.suggest_categorical('weight_decay', search_space['weight_decay']), 
              'gamma': trial.suggest_categorical('gamma', search_space['gamma']), 
              'fnn_n_fc': trial.suggest_categorical('fnn_n_fc', search_space['fnn_n_fc']), 
              'fnn_dim': trial.suggest_categorical('fnn_dim', search_space['fnn_dim'])}
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
