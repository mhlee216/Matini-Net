import sys
sys.path.append("/home/mhlee/MatiniNet/")
import matini_net.database as midb
import matini_net.data as midt

from matminer.datasets import load_dataset
import pandas as pd
from tqdm.notebook import tqdm
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='matbench_phonons')
parser.add_argument('--poscar_path', type=str, default='./poscar/')
parser.add_argument('--pbar', type=str, default='True')
parser.add_argument('--timeout', type=float, default=60*5)
parser.add_argument('--num_sep', type=int, default=1000)
parser.add_argument('--output', type=str, default='./matbench_phonons')
args = parser.parse_args()
        
if args.pbar == 'True':
    args.pbar = True
else:
    args.pbar = False

df = load_dataset(args.input)
df['id'] = [n for n in range(1, df.shape[0]+1)]

# saving structures as posca files
df['poscar'] = 0
for i in tqdm(range(df.shape[0]), desc='Loading poscar'):
    file_name = f"{args.poscar_path}{args.input}_{df['id'].iloc[i]}.POSCAR"
    try:
        midb.save_poscar(df['structure'].iloc[i], file_name)
        df['poscar'].iloc[i] = file_name
    except:
        print('not saved:', file_name)
df = df[df['poscar'] != 0].reset_index(drop=True)
df.to_csv(args.input+'.csv', index=False)

if args.num_sep == 'None':
    num_sep = df.shape[0]
else:
    num_sep = args.num_sep
    
n, i = 1, 0
total_shape = pd.read_csv(args.input+'.csv').shape[0]
with pd.read_csv(args.input+'.csv', chunksize=num_sep) as reader:
    for df in reader:
        if i+num_sep > total_shape:
            start, end = i, total_shape
        else:
            start, end = i, i+num_sep
        i += num_sep
        print(n, start, end)
        
        for n in tqdm(range(df.shape[0]), desc='Loading structures'):
            df['structure'].iloc[n] = midb.load_poscar(df['poscar'].iloc[n])
        
        print('Converting')
        print('get_composition')
        df = midt.get_composition(df, pbar=args.pbar)
        print(':', df.shape)
        df = df.dropna(axis=0).reset_index(drop=True)
        print('convert_composition')
        df = midt.convert_composition(df, pbar=args.pbar)
        print(':', df.shape)
        
        print('Featurizing')
        print('composition_featurizer')
        df = midt.composition_featurizer(df, pbar=args.pbar)
        print(':', df.shape)
        print('site_featurizer')
        df = midt.site_featurizer(df, pbar=args.pbar)
        print(':', df.shape)
        print('structure_featurizer')
        df = midt.structure_featurizer(df, pbar=args.pbar)
        print(':', df.shape)
        
        print('Saving')
        del df['structure']
        print(':', df.shape)
        df.to_csv(f'{args.output}_{start}_{end}.csv', index=False)
        n += 1
