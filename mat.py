import os

# os.system('conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch -y')

os.system('pip install torch torchvision torchaudio')
os.system('pip install torch_geometric')
os.system('pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html')

os.system('pip install notebook')
os.system('pip install ipywidgets')
os.system('pip install ipython')
# os.system('conda install pyg -c pyg -y')
# os.system('pip install torch-geometric==1.7.1')

os.system('pip install pymatgen')
os.system('pip install matminer')
os.system('pip install mdf-forge')
os.system('pip install qmpy-rester')
os.system('pip install ase')
os.system('pip install shap')
os.system('pip install parmap')
os.system('pip install matbench')
os.system('pip install scikit-learn==1.0.1')
os.system('pip install scipy==1.7.3')
os.system('pip install numpy==1.22.0') 
os.system('pip install optuna')

os.system('conda install -c conda-forge rdkit -y')

