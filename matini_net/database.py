#!/usr/bin/env python
# coding: utf-8
# database.py

import io
import json
import pandas as pd
import numpy as np
import requests
import qmpy_rester as qr
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import IStructure
import warnings
warnings.filterwarnings('ignore')




# The Open Quantum Materials Database (OQMD)

def convert_oqmd_poscar(data):
    oqmd_id = int(data['entry_id'])
    icsd_id = int(data['icsd_id'])
    atoms = {}
    for s in data['sites']:
        ele = s.strip().split('@')[0].strip()
        pos = s.strip().split('@')[1].strip()
        if ele in atoms:
            atoms[ele].append(pos)
        else:
            atoms[ele] = [pos]
    poscar = []
    poscar.append('oqmd_id_%d, icsd_id_%d\n'%(oqmd_id,icsd_id))
    poscar.append('1.0\n')
    for v in data['unit_cell']:
        poscar.append('%.3f %.3f %.3f\n'%tuple(v))
    atom_lst = atoms.keys()
    poscar.append('%s\n'%('  '.join(atom_lst)))
    poscar.append('%s\n'%('  '.join([str(len(atoms[v])) for v in atom_lst])))
    poscar.append('Direct\n')
    for a in atom_lst:
        poscar.append('%s\n'%('\n'.join(atoms[a])))
    return ''.join(poscar)

def get_oqmd_props(oqmd_ids, dropna=True):
    data = []
    for entry_id in oqmd_ids:
        oqmd_api = requests.get(f"https://oqmd.org/oqmdapi/entry/{entry_id}/")
        data.append(oqmd_api.json())
    df = pd.DataFrame(data)
    df = df.rename(columns={'id':'entry_id'})
    data = json.loads(df.to_json(orient='records'))
    df['structure'] = 0
    for i in range(len(data)):
        try:
            poscar = convert_oqmd_poscar(data[i])
            df['structure'].iloc[i] = IStructure.from_str(poscar, fmt='POSCAR')
        except:
            df['structure'].iloc[i] = np.nan
    df = df[['entry_id', 'name', 'band_gap', 'volume', 'composition', 'structure']]
    df = df.rename(columns={'entry_id':'id'})
    df = df.rename(columns={'name':'formula'})
    if dropna:
        df = df.dropna(axis=0).reset_index(drop=True)
    return df

def phase_oqmd_props(phases, dropna=True):
    with qr.QMPYRester() as q:
        phased_data = q.get_oqmd_phases(**phases)
    data = phased_data['data']
    df = pd.DataFrame(data)
    df['structure'] = 0
    for i in range(len(data)):
        try:
            poscar = convert_oqmd_poscar(data[i])
            df['structure'].iloc[i] = IStructure.from_str(poscar, fmt='POSCAR')
        except:
            df['structure'].iloc[i] = np.nan
    df = df[['entry_id', 'name', 'band_gap', 'volume', 'composition', 'structure']]
    df = df.rename(columns={'entry_id':'id'})
    df = df.rename(columns={'name':'formula'})
    if dropna:
        df = df.dropna(axis=0).reset_index(drop=True)
    return df

def save_oqmd_poscar(structure_info, file_name):
    if '.POSCAR' in file_name.upper():
        f = open(file_name.replace('.poscar', '.POSCAR'), 'w')
        f.write(str(structure_info))
        f.close()
    else:
        raise ValueError("The file name is not in .POSCAR format.")




# The Materials Project (MP)

def get_mp_props(api_key, mp_ids, dropna=True):
    mp_props=['material_id', 
              'formula_pretty', 
              'composition', 
              'structure', 
              'band_gap', 
              'volume']
    mpr = MPRester(api_key)
    data = mpr.summary.search(material_ids=mp_ids, fields=mp_props)
    material_ids = list(set([str(data[i].material_id) for i in range(len(data))]))
    total_data_list = []
    for i in range(len(data)):
        data_list = []
        for prop in mp_props:
            try:
                data_list.append(data[i].dict()[prop])
            except:
                data_list.append(np.nan)
        total_data_list.append(data_list)
    df = pd.DataFrame(total_data_list, columns=mp_props)
    df = df[['material_id', 
             'formula_pretty', 
             'band_gap', 
             'volume', 
             'composition', 
             'structure']]
    df = df.rename(columns={'material_id':'id'})
    df = df.rename(columns={'formula_pretty':'formula'})
    if dropna:
        df = df.dropna(axis=0).reset_index(drop=True)
    return df




# Utility

def save_poscar(structure_info, file_name):
    if '.POSCAR' in file_name.upper():
        structure_info.to(file_name.replace('.poscar', '.POSCAR'))
    else:
        raise ValueError("The file name is not in .POSCAR format.")

def load_poscar(file_name):
    if '.POSCAR' in file_name.upper():
        file_name = file_name.replace('.poscar', '.POSCAR')
        structure_info = IStructure.from_file(file_name)
        return structure_info
    else:
        raise ValueError("The file name is not in .POSCAR format.")
