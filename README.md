# FG-DDI: 

**Authors**: Fangyu Zhou, Shahadat Uddin 

# Requirement
To run the code, you need the following dependencies:
* Python == 3.7
* PyTorch == 1.9.0
* PyTorch Geometry == 2.0.3
* rdkit == 2020.09.2

# Installation
You can create a virtual environment using conda 
```bash
conda create -n FG-DDI python=3.7
conda activate FG-DDI
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==2.0.3
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
conda install -c rdkit rdkit
```

# Dataset
The datasete were sourced from the baseline model DSN-DDI.
```


