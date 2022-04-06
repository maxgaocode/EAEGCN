    



python train.py   --dataset cora  --weight_decay 1E-3  --hidden 16  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.2   --layer_num   4  --train_ratio  0.6   --patience   100
python train.py   --dataset citeseer  --weight_decay 1E-3  --hidden 16  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.3   --layer_num   4  --train_ratio  0.6   --patience   100
python train.py   --dataset pubmed  --weight_decay 1E-3  --hidden 16  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.3   --layer_num   4  --train_ratio  0.6   --patience   100



python train.py   --dataset cornell --weight_decay 5E-5  --hidden 64  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.4   --layer_num   2  --train_ratio  0.6   --patience   200
python train.py   --dataset texas  --weight_decay 5E-5  --hidden 64  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.4   --layer_num   2  --train_ratio  0.6   --patience   200
python train.py   --dataset wisconsin --weight_decay 5E-5  --hidden 64  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.4   --layer_num   2  --train_ratio  0.6   --patience   200

Actor or film 
python train.py   --dataset film  --weight_decay 5E-5  --hidden 64  --epochs  800  --dropout  0.5  --dropedge  0.5  --eps  0.4   --layer_num  2  --train_ratio  0.6   --patience   200




