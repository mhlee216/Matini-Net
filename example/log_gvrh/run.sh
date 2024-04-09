#!/bin/bash

python matini_net_gnn.py --gnn 'cgcnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_gnn.py --gnn 'gcn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_gnn.py --gnn 'megnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_gnn.py --gnn 'mpnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_gnn.py --gnn 'schnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_fnn.py --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_fnn_gf.py --gnn 'cgcnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf.py --gnn 'gcn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf.py --gnn 'megnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf.py --gnn 'mpnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf.py --gnn 'schnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_direct.py --gnn 'cgcnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct.py --gnn 'gcn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct.py --gnn 'megnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct.py --gnn 'mpnn' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct.py --gnn 'schnet' --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_multi.py --gnn 'cgcnn' --att 0 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_multi.py --gnn 'gcn' --att 0 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_multi.py --gnn 'megnet' --att 0 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_multi.py --gnn 'mpnn' --att 0 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_multi.py --gnn 'schnet' --att 0 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
