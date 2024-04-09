#!/bin/bash

python matini_net_direct_shap.py --case 'direct_cgcnn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct_shap.py --case 'direct_gcn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct_shap.py --case 'direct_megnet' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct_shap.py --case 'direct_mpnn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_direct_shap.py --case 'direct_schnet' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_fnn_shap.py --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_shap.py --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_shap.py --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_shap.py --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_shap.py --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'

python matini_net_fnn_gf_shap.py --case 'fnn_gf_cgcnn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf_shap.py --case 'fnn_gf_gcn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf_shap.py --case 'fnn_gf_megnet' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf_shap.py --case 'fnn_gf_mpnn' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
python matini_net_fnn_gf_shap.py --case 'fnn_gf_schnet' --k 1 --device 'cuda:0' --target 'log10(G_VRH)' --nan 'dropna'
