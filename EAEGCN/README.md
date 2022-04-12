# EAEGCN

This is the source code for our paper: [Edge-oriented Attention Mechanism for Graph Convolution Networks]

--------------------------------------------------------------------------------

## Requirements

+ python==3.6, 3.7
+ dgl>=0.6.0
+ torch_sparse

## Usage

### Datas


### From source


Then run:

```
the model should install the cuda version of dgl
pip install dgl-cu101
```

When running in a docker container without nvidia driver, PyTorch needs to evaluate the compute capabilities and may fail.
In this case,  *e.g.*:

```
cd EAEGCN
source run.sh
```
 

Results
```
|             | `cora` | `citeseer` | `pubmed` | 
|-------------|-------|------------|----------|
| **ours**   |84.5    |   72.7     |     80.6   |    

```





You  can train the model with more parameters for 

python python mainepoch.py   --dataset [] --weight_decay []  --hidden []  --epochs  [] --dropedge  [] --eps  []   --layer_num  []  --train_ratio  []   --patience  []



## Running Environment 

The experimental results reported in paper are conducted on a single NVIDIA GeForce RTX 1080 Ti with CUDA 10.1, which might be slightly inconsistent with the results induced by other platforms.

