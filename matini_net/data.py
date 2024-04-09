#!/usr/bin/env python
# coding: utf-8
# data.py

import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import multiprocessing
from matminer.featurizers import conversions # Conversion utilities.
from matminer.featurizers import composition # Features based on a material’s composition.
from matminer.featurizers import site # Features from individual sites in a material’s crystal structure.
from matminer.featurizers import structure # Generating features based on a material’s crystal structure.
from pymatgen.analysis.local_env import VoronoiNN
import ase
from ase import io
from scipy.stats import rankdata
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse, add_self_loops
import warnings
warnings.filterwarnings('ignore')




# Featurizer

def featurizing_with_timeout(func, args, timeout):
    result_queue = multiprocessing.Queue()
    def wrapper_function():
        result = func(*args)
        result_queue.put(result)
    process = multiprocessing.Process(target=wrapper_function)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    if not result_queue.empty():
        result = result_queue.get()
        return result

def featurizing(featurizer, df, column):
    df = pd.DataFrame(df[column])
    df = featurizer.featurize_dataframe(df, column, ignore_errors=True, pbar=False)
    df = df.drop([column], axis=1)
    return df

def get_composition(df, col='structure', pbar=False):
    ft = conversions.StructureToComposition()
    df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    return df

def convert_composition(df, col='composition', pbar=False):
    if col == 'composition':
        df['str_composition'] = df['composition']
        del df['composition']
        ft = conversions.StrToComposition()
        df = ft.featurize_dataframe(df, 'str_composition', ignore_errors=True, pbar=pbar)
        del df['str_composition']
    else:
        ft = conversions.StrToComposition()
        df = ft.featurize_dataframe(df, col, ignore_errors=True)
        del df[col]
    return df

def convert_composition_oxid(df, col='composition', pbar=False):
    ft = conversions.CompositionToOxidComposition()
    df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    return df

def convert_structure_oxid(df, col='structure', timeout=60*1, pbar=False):
#     ft = conversions.StructureToOxidStructure()
#     df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    ft = conversions.StructureToOxidStructure()
    small_df = pd.DataFrame()
    if pbar:
        rng = tqdm(range(len(df)), desc="StructureToOxidStructure")
    else:
        rng = range(len(df))
    for i in rng:
        df_small = df.iloc[[i]]
        result_feat = featurizing_with_timeout(featurizing, (ft, df_small, col), timeout)
        if result_feat is None:
            result_feat = pd.DataFrame([np.nan])
        else:
            pass
        if result_feat.shape[1] >= small_df.shape[1]:
            small_df = pd.concat([result_feat, small_df.reindex(columns=result_feat.columns)], axis=0, ignore_index=True)
        else:
            small_df = pd.concat([result_feat.reindex(columns=small_df.columns), small_df], axis=0, ignore_index=True)
    df = pd.concat([df, small_df], axis=1).reset_index(drop=True)
    return df

def composition_featurizer(df, col='composition', fast=True, pbar=False):
    # # composition
    # # Features based on a material’s composition.
    for ft in [composition.Miedema(), 
               composition.ElementProperty.from_preset(preset_name='magpie'), 
               composition.ElementFraction(), 
               composition.TMetalFraction(), 
               composition.Stoichiometry(), 
               composition.BandCenter(), 
               composition.AtomicOrbitals(), 
               composition.ValenceOrbital(props=['frac'])]:
        df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    df = df.drop(['HOMO_element', 'LUMO_element', 'HOMO_character', 'LUMO_character', 
                  'Lr', 'No', 'Fm', 'Fr', 'Ra', 'Po', 'Ne', 'He', 'Xe', 'Ar', 
                  'Md', 'Am', 'Cm', 'Bk', 'Cf', 'Kr', 'At', 'Rn', 'Es'], axis=1)
    if not fast:
        for ft in [composition.WenAlloys(), 
                   composition.AtomicPackingEfficiency()]:
            df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
        df = df.drop(['Weight Fraction', 'Atomic Fraction', 
                      'mean simul. packing efficiency', 'mean abs simul. packing efficiency'], axis=1)
    return df

def composition_oxid_featurizer(df, col='composition_oxid', fast=True, pbar=False):
    # composition
    # Features based on a material’s composition.
    for ft in [composition.IonProperty(fast=False), 
               composition.OxidationStates()]:
        df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    return df

def site_featurizer(df, col='structure', fast=True, pbar=False):
    # site
    # Features from individual sites in a material’s crystal structure.
    for site_ft in [site.AverageBondAngle(VoronoiNN()), 
                    site.AverageBondLength(VoronoiNN()), 
                    site.AGNIFingerprints(), 
                    site.CrystalNNFingerprint.from_preset(preset='cn'),
                    site.VoronoiFingerprint(),
                    site.CoordinationNumber.from_preset(preset="VoronoiNN"), 
                    site.IntersticeDistribution(), 
                    site.GaussianSymmFunc(), 
                    site.GeneralizedRadialDistributionFunction.from_preset("gaussian")]:
        ft = structure.sites.SiteStatsFingerprint(site_ft)
        df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    df = df.drop(["mean Interstice_dist_mean", "mean Interstice_dist_maximum", 
                  "mean Interstice_dist_std_dev", "mean Interstice_dist_minimum", 
                  "std_dev Interstice_dist_std_dev", "std_dev Interstice_dist_mean", 
                  "std_dev Interstice_dist_minimum", "std_dev Interstice_dist_maximum"], axis=1)
    if not fast:
        for site_ft in [site.BondOrientationalParameter(), 
                        site.LocalPropertyDifference(), 
                        site.OPSiteFingerprint()]:
            ft = structure.sites.SiteStatsFingerprint(site_ft)
            df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    return df

def structure_featurizer(df, col='structure', col_oxid='structure_oxid', fast=True, pbar=False, timeout=60*1):
    # structure
    # Generating features based on a material’s crystal structure.
    for ft in [structure.XRDPowderPattern(), 
               structure.ChemicalOrdering(), 
               structure.DensityFeatures(), 
               structure.MaximumPackingEfficiency(), 
               structure.StructuralComplexity(), 
               structure.RadialDistributionFunction(), 
               structure.Dimensionality(), 
               structure.GlobalSymmetryFeatures()]:
        df = ft.featurize_dataframe(df, col, ignore_errors=True, pbar=pbar)
    df = df.drop(['crystal_system'], axis=1)
    if not fast:
        ##
        ft = structure.StructuralHeterogeneity()
        small_df = pd.DataFrame()
        if pbar:
            rng = tqdm(range(len(df)), desc="StructuralHeterogeneity")
        else:
            rng = range(len(df))
        for i in rng:
            df_small = df.iloc[[i]]
            result_feat = featurizing_with_timeout(featurizing, (ft, df_small, col), timeout)
            if result_feat is None:
                result_feat = pd.DataFrame([np.nan])
            else:
                pass
            if result_feat.shape[1] >= small_df.shape[1]:
                small_df = pd.concat([result_feat, small_df.reindex(columns=result_feat.columns)], axis=0, ignore_index=True)
            else:
                small_df = pd.concat([result_feat.reindex(columns=small_df.columns), small_df], axis=0, ignore_index=True)
        df = pd.concat([df, small_df], axis=1).reset_index(drop=True)
        ##
        ft = structure.JarvisCFID()
        small_df = pd.DataFrame()
        if pbar:
            rng = tqdm(range(len(df)), desc="JarvisCFID")
        else:
            rng = range(len(df))
        for i in rng:
            df_small = df.iloc[[i]]
            result_feat = featurizing_with_timeout(featurizing, (ft, df_small, col), timeout)
            if result_feat is None:
                result_feat = pd.DataFrame([np.nan])
            else:
                pass
            if result_feat.shape[1] >= small_df.shape[1]:
                small_df = pd.concat([result_feat, small_df.reindex(columns=result_feat.columns)], axis=0, ignore_index=True)
            else:
                small_df = pd.concat([result_feat.reindex(columns=small_df.columns), small_df], axis=0, ignore_index=True)
        df = pd.concat([df, small_df], axis=1).reset_index(drop=True)
        df = df.drop(['jml_C-6', 'jml_C-12', 'jml_C-13', 'jml_C-18', 'jml_C-19', 
                      'jml_C-20', 'jml_C-24', 'jml_C-25', 'jml_C-26', 'jml_C-27', 
                      'jml_C-30', 'jml_C-31', 'jml_C-32', 'jml_C-33', 'jml_C-34', 
                      'jml_is_noble_gas'], axis=1)
#         ##
#         ft = structure.ElectronicRadialDistributionFunction()
#         small_df = pd.DataFrame()
#         if pbar:
#             rng = tqdm(range(len(df)), desc="ElectronicRadialDistributionFunction")
#         else:
#             rng = range(len(df))
#         for i in rng:
#             df_small = df.iloc[[i]]
#             result_feat = featurizing_with_timeout(featurizing, (ft, df_small, col_oxid), timeout)
#             if result_feat is None:
#                 result_feat = pd.DataFrame([np.nan])
#             else:
#                 pass
#             if result_feat.shape[1] >= small_df.shape[1]:
#                 small_df = pd.concat([result_feat, small_df.reindex(columns=result_feat.columns)], axis=0, ignore_index=True)
#             else:
#                 small_df = pd.concat([result_feat.reindex(columns=small_df.columns), small_df], axis=0, ignore_index=True)
#         df = pd.concat([df, small_df], axis=1).reset_index(drop=True)
    return df

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_pretrained_graph_featurizer(graph_nn):
    graph_nn.model.lin_out = Identity()
    return graph_nn

def graph_featurizer(graph_nn, data_list, batch_size, device):
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, drop_last=False)
    data_list = []
    graph_nn = graph_nn.to(device)
    graph_nn.eval()
    graph_nn.gf = True
    for data in data_loader:
        data = data.to(device)
        x = graph_nn(data).detach().to('cpu').tolist()
        data_list.extend(x)
    return torch.tensor(data_list)


# Graph

eye = np.eye(100)
atom_dict = {}
for i in range(100):
    atom_dict[str(i+1)] = eye[i].astype(int).tolist()

def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed, method="ordinal", axis=1)
    elif reverse == True:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed * -1, method="ordinal", axis=1)
    distance_matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, distance_matrix_trimmed))
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0
    if adj == False:
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(temp, 
                                    pad_width=(0, neighbors + 1 - len(temp)), 
                                    mode="constant", 
                                    constant_values=0)
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed, adj_list, adj_attr

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)
    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def get_graph_from_poscar(poscar_file, threshold=8.0, neighbors=12, reverse=False, adj=False):
    data = Data()
    distance_gaussian = GaussianSmearing(0, 1, 50, 0.2)
    ase_crystal = ase.io.read(poscar_file)
    data.ase = ase_crystal
    distance_matrix = ase_crystal.get_all_distances(mic=True)
    distance_matrix_trimmed = threshold_sort(distance_matrix, 
                                             threshold=threshold, 
                                             neighbors=neighbors, 
                                             reverse=reverse, 
                                             adj=adj)
    distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
    out = dense_to_sparse(distance_matrix_trimmed)
    edge_index, edge_weight = add_self_loops(out[0], out[1], num_nodes=len(ase_crystal), fill_value=0)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long)
    data.edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    data.edge_attr = torch.tensor(distance_gaussian(edge_weight), dtype=torch.float)
    distance_matrix_mask = (distance_matrix_trimmed.fill_diagonal_(1) != 0).int()
    z = torch.LongTensor(ase_crystal.get_atomic_numbers())
    data.z = z
    atom_fea = np.vstack([atom_dict[str(data.ase.get_atomic_numbers()[i])] 
                          for i in range(len(data.ase))]).astype(float)
    data.x = torch.tensor(atom_fea, dtype=torch.float)
    return data


# Data

def split_feature_df(df, drop_cols):
    inf_df = df[drop_cols]
    feature_df = df.drop(drop_cols, axis=1)
    for col in ['id', 'formula', 'structure', 'composition', 'composition_oxid', 'poscar']:
        if col in feature_df.columns:
            feature_df = feature_df.drop(col, axis=1)
    return inf_df, feature_df

def get_shap_selected_features(df, feature_df, shap_values, feat_num=None, cutoff=None):
    if feat_num != None and cutoff != None:
        raise ValueError("Set only one of 'feat_num' and 'cutoff'.")
    selected_feature_df = pd.DataFrame(shap_values).iloc[:, -feature_df.shape[1]:]
    cols = feature_df.columns
    selected_feature_df.columns = cols
    selected_feature_df = pd.DataFrame({'Feature':cols, 'AbsMax':[selected_feature_df[c].abs().max() for c in cols]})
    if feat_num != None:
        selected_feature_df = selected_feature_df.sort_values(['AbsMax'], ascending=False).iloc[:feat_num]
    elif cutoff != None:
        selected_feature_df = selected_feature_df[selected_feature_df['AbsMax'] > cutoff]
    else:
        raise ValueError("Set either 'feat_num' or 'cutoff'.")
    df = df[selected_feature_df['Feature'].tolist()]
    return df

def get_data_list(inf_df, feature_df, y, poscar_path_col='poscar'):
    data_list = []
    for i in tqdm(range(inf_df.shape[0])):
        file_name = inf_df[poscar_path_col].iloc[i]
        data = get_graph_from_poscar(file_name)
        data.feat = torch.tensor([feature_df.iloc[i].values.tolist()], dtype=torch.float)
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u
        data.y = torch.tensor([inf_df[y].iloc[i]], dtype=torch.float)
        data_list.append(data)
    return data_list

def get_graph_features(data_list, graph_nn, batch_size, device):
    graph_feature = graph_featurizer(graph_nn, data_list, batch_size, device)
    graph_feature = graph_feature.reshape(-1, 1, graph_feature.shape[-1])
    for data, graph_feat in tqdm(zip(data_list, graph_feature)):
        data.feat = torch.concat((graph_feat, data.feat), axis=1)
    return data_list
