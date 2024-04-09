#!/usr/bin/env python
# coding: utf-8
# process.py

import torch
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import json
import copy
import shap
import warnings
warnings.filterwarnings('ignore')




class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def dataloader(data_list, batch_size, shuffle=False, drop_last=False):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train(model, epochs, train_loader, criterion, optimizer, scheduler, device):
    model = model.to(device)
    train_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        labels = data.y
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.to('cpu').reshape(-1), 
                         labels.to('cpu').reshape(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return model, train_loss

def valid(model, valid_loader, criterion, device):
    model = model.to(device)
    valid_loss = 0
    model.eval()
    for data in valid_loader:
        data = data.to(device)
        labels = data.y
        outputs = model(data)
        loss = criterion(outputs.to('cpu').reshape(-1), 
                         labels.to('cpu').reshape(-1))
        valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    return valid_loss

def test(model, test_loader, device):
    model = model.to(device)
    true = []
    pred = []
    model.eval()
    for data in test_loader:
        data = data.to(device)
        labels = data.y
        outputs = model(data)
        true.extend(labels.to('cpu').reshape(-1).tolist())
        pred.extend(outputs.to('cpu').reshape(-1).tolist())
    return true, pred

def experiment(model, epochs, train_loader, valid_loader, criterion, optimizer, scheduler, early_stopper, device, prints=True):
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        model, train_loss = train(model, epochs, train_loader, criterion, optimizer, scheduler, device)
        valid_loss = valid(model, valid_loader, criterion, device)
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        if prints:
            print('Epoch: {}/{}\t Train Loss: {:.4f}\t Test Loss: {:.4f}'.format(epoch+1, epochs, train_loss, valid_loss))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        if early_stopper.early_stop(valid_loss):             
            if prints:
                print('Stop!')
            break
    if prints:
        print('Done!')
        print()
    return model, train_loss_list, valid_loss_list

def predict(model, test_loader, device):
    pred = []
    model.eval()
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        pred.extend(outputs.to('cpu').reshape(-1).tolist())
    return pred

def model_save(model, path):
    torch.save(model, path)
    
def model_load(path):
    return torch.load(path)

def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

def loss_plot(train_loss_list, test_loss_list, 
              figsize=(4, 4), dpi=None, 
              xlim=None, ylim=None, 
              train_c='b', test_c='r', 
              xlabel='Epochs', ylabel='Loss', 
              xlabel_fs=18, ylabel_fs=18, 
              xticks_fs=16, yticks_fs=16, 
              legend_fs=16, show=True):
    if dpi == None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=figsize, dpi=dpi)
    plt.plot([i for i in range(len(train_loss_list))], train_loss_list, c=train_c, label='Train Loss')
    plt.plot([i for i in range(len(test_loss_list))], test_loss_list, c=test_c, label='Valid Loss')
    plt.xlabel('Epochs', fontsize=xlabel_fs)
    plt.ylabel('Loss', fontsize=ylabel_fs)
    plt.xticks(fontsize=xticks_fs)
    plt.yticks(fontsize=yticks_fs)
    plt.legend(fontsize=legend_fs)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if show:
        plt.show()

def regre_scores(true, pred):
    pcc = pearsonr(true, pred)[0]
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    return pcc, r2, mae, mse

def regre_plot(true, pred, 
               figsize=(4, 4), dpi=None, 
               xlim=None, ylim=None, 
               c='b', a=0.6, s=5, 
               xlabel='True', ylabel='Pred', 
               xlabel_fs=18, ylabel_fs=18, 
               xticks_fs=16, yticks_fs=16, 
               show=True, prints=True):
    pcc, r2, mae, mse = regre_scores(true, pred)
    vs = true + pred
    if dpi == None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=figsize, dpi=dpi)
    if xlim != None and ylim != None:
        plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], c='gray', alpha=0.8)
    else:
        plt.plot([min(vs), max(vs)], [min(vs), max(vs)], c='gray', alpha=0.8)
    plt.scatter(true, pred, c=c, alpha=a, s=s)
    plt.xlabel('True', fontsize=xlabel_fs)
    plt.ylabel('Pred', fontsize=ylabel_fs)
    plt.xticks(fontsize=xticks_fs)
    plt.yticks(fontsize=yticks_fs)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if show:
        plt.show()
    if prints:
        print('- PCC : %.4f' % pcc)
        print('- R2 : %.4f' % r2)
        print('- MAE : %.4f' % mae)
        print('- MSE : %.4f' % mse)
    return pcc, r2, mae, mse



# Optuna
def get_n_trials(search_space):
    n_trial_list = [len(search_space[k]) for k in search_space.keys()]
    n_trials = 1
    for n in n_trial_list:
        n_trials *= n
    return n_trials

def optuna_cross_validation(params, model, train_data_list, prints=True, optuna=True, save_model=True, path='model'):
    folds = KFold(n_splits=params['n_splits'], shuffle=True, random_state=params['seed'])
    model_list = []
    train_loss_list = []
    valid_loss_list = []
    true_list = []
    pred_list = []
    score_list = []
    for k, (train_idx , valid_idx) in enumerate(folds.split(train_data_list)):
        if prints:
            print("- Fold: {}".format(k+1))
        train_data = [x for i, x in enumerate(train_data_list) if i in train_idx]
        valid_data = [x for i, x in enumerate(train_data_list) if i in valid_idx]
        if prints:
            print("- Train: {}".format(len(train_data)))
            print("- Valid: {}".format(len(valid_data)))
        train_loader = dataloader(train_data, batch_size=params['batch'], shuffle=True, drop_last=False)
        valid_loader = dataloader(valid_data, batch_size=params['batch'], shuffle=True, drop_last=False)
        criterion = torch.nn.MSELoss()
        m = copy.deepcopy(model)
        optimizer = torch.optim.Adam(m.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
        early_stopper = EarlyStopper(patience=params['patience'])
        trained_model, train_loss, valid_loss = experiment(m, 
                                                           params['epochs'], 
                                                           train_loader, 
                                                           valid_loader, 
                                                           criterion, 
                                                           optimizer, 
                                                           scheduler, 
                                                           early_stopper, 
                                                           params['device'], 
                                                           prints)
        true, pred = test(trained_model, valid_loader, params['device'])
        pcc, r2, mae, mse = regre_scores(true, pred)
        score = mae
        model_list.append(trained_model)
        if save_model:
            model_save(trained_model, f'{path}_{k+1}.pt')
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        true_list.append(true)
        pred_list.append(pred)
        score_list.append(score)
    cv_score = np.array(score_list).mean()
    if optuna:
        return cv_score
    else:
        return model_list, train_loss_list, valid_loss_list, true_list, pred_list

def optuna_train_test_split(params, model, train_data_list, prints=True, optuna=True, save_model=True, path='model.pt'):
    train_data, valid_data = train_test_split(train_data_list, test_size=params['test_size'], random_state=params['seed'])
    if prints:
        print("- Train: {}".format(len(train_data)))
        print("- Valid: {}".format(len(valid_data)))
    train_loader = dataloader(train_data, batch_size=params['batch'], shuffle=True, drop_last=False)
    valid_loader = dataloader(valid_data, batch_size=params['batch'], shuffle=True, drop_last=False)
    criterion = torch.nn.MSELoss()
    m = copy.deepcopy(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    early_stopper = EarlyStopper(patience=params['patience'])
    trained_model, train_loss, valid_loss = experiment(m, 
                                                       params['epochs'], 
                                                       train_loader, 
                                                       valid_loader, 
                                                       criterion, 
                                                       optimizer, 
                                                       scheduler, 
                                                       early_stopper, 
                                                       params['device'], 
                                                       prints)
    true, pred = test(trained_model, valid_loader, params['device'])
    pcc, r2, mae, mse = regre_scores(true, pred)
    score = mae
    if save_model:
        model_save(trained_model, path)
    if optuna:
        return score
    else:
        return trained_model, train_loss, valid_loss, true, pred

def optuna_train_test(params, model, train_data, valid_data, prints=True, optuna=True, save_model=True, path='model.pt'):
    if prints:
        print("- Train: {}".format(len(train_data)))
        print("- Valid: {}".format(len(valid_data)))
    train_loader = dataloader(train_data, batch_size=params['batch'], shuffle=True, drop_last=False)
    valid_loader = dataloader(valid_data, batch_size=params['batch'], shuffle=True, drop_last=False)
    criterion = torch.nn.MSELoss()
    m = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    early_stopper = EarlyStopper(patience=params['patience'])
    trained_model, train_loss, valid_loss = experiment(m, 
                                                       params['epochs'], 
                                                       train_loader, 
                                                       valid_loader, 
                                                       criterion, 
                                                       optimizer, 
                                                       scheduler, 
                                                       early_stopper, 
                                                       params['device'], 
                                                       prints)
    true, pred = test(trained_model, valid_loader, params['device'])
    pcc, r2, mae, mse = regre_scores(true, pred)
    score = mae
    if save_model:
        model_save(trained_model, path)
    if optuna:
        return score
    else:
        return trained_model, train_loss, valid_loss, true, pred

def save_param(file_name, study):
    params = study.best_trial.params
    params['best_score'] = study.best_trial.value
    with open(file_name, 'w') as f:
        json.dump(params, f)
    return params

def load_param(file_name):
    with open(file_name, 'r') as f:
        params = json.load(f)
    return params

def regre_save_json(file_name, pcc, r2, mae, mse, true, pred, train_loss=None, valid_loss=None):
    if train_loss != None and valid_loss != None:
        result = pd.DataFrame({'pcc':[pcc], 
                               'r2':[r2], 
                               'mae':[mae], 
                               'mse':[mse], 
                               'true':[true], 
                               'pred':[pred], 
                               'train_loss':[train_loss], 
                               'valid_loss':[valid_loss]})
    else:
        result = pd.DataFrame({'pcc':[pcc], 
                               'r2':[r2], 
                               'mae':[mae], 
                               'mse':[mse], 
                               'true':[true], 
                               'pred':[pred]})
    result.to_json(file_name, orient='table')
    return result




# SHAP
def shap_tensor(graph_nn, feature_nn, data_list, device, batch_size=128, shuffle=False, drop_last=False):
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    data_x_list = []
    for data in data_loader:
        data = data.to(device)
        gx = graph_nn(data)
        fx = feature_nn(data)
        x = torch.concat((gx, fx), axis=1).detach().to('cpu').tolist()
        data_x_list.extend(x)
    return torch.tensor(data_x_list).to(device)

def shap_tensor_single(data_list, device, batch_size=128, shuffle=False, drop_last=False):
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    data_x_list = []
    for data in data_loader:
        x = data.feat.detach().to('cpu').tolist()
        data_x_list.extend(x)
    return torch.tensor(data_x_list).to(device)

def shap_explainer(concat_nn, train_x_list, test_x_list):
    e = shap.DeepExplainer(concat_nn, train_x_list)
    shap_values = e.shap_values(test_x_list)
    return e, shap_values

def save_shap_data(file_name, shap_values, data_values):
    shap_data = {'shap_values':shap_values.tolist(), 
                 'data_values':data_values.to('cpu').numpy().tolist()}
    with open(file_name, 'w') as f:
        json.dump(shap_data, f)

def load_shap_data(file_name):
    with open(file_name, 'r') as f:
        shap_data = json.load(f)
    shap_values = np.array(shap_data['shap_values'])
    data_values = np.array(shap_data['data_values'])
    return shap_values, data_values
