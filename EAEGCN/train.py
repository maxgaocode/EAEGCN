import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from utilsmix import accuracy, preprocess_data
from model import MAGCN
import warnings
warnings.filterwarnings('ignore')
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5E-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate in hidden.')
parser.add_argument('--dropedge', type=float, default=0.5, help='Dropout rate in edge.')
parser.add_argument('--eps', type=float, default=0.4, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()


device = 'cuda'

g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.train_ratio)




features = features.to(device)
labels = labels.to(device)
rain = train.to(device)
test = test.to(device)
val = val.to(device)
g = g.to(device)

deg = g.in_degrees().cuda().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm

net = MAGCN(g, features.size()[1], args.hidden, nclass, args.dropout, args.dropedge, args.eps, args.layer_num).cuda()

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# main loop
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_acc = 0.0

for epoch in range(args.epochs):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    output = net(features)

    cla_loss = F.nll_loss(output[train], labels[train])
    loss = cla_loss
    train_acc = accuracy(output[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    output = net(features)
    test_acc = accuracy(output[test], labels[test])
    loss_val = F.nll_loss(output[val], labels[val]).item()
    val_acc = accuracy(output[val], labels[val])
    los.append([epoch, loss_val, val_acc, test_acc])

    if loss_val < min_loss and max_acc < val_acc:
        min_loss = loss_val
        max_acc = val_acc
        counter = 0
    else:
        counter += 1

    if counter >= args.patience and args.dataset in ['cora', 'citeseer', 'pubmed']:
        print('early stop')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)
        if epoch % 20== 0:
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))


if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
    los.sort(key=lambda x: x[1])
    acc = los[0][-1]
    print("the process is in the acc print foot")
    print(type(acc))
    print(acc)
else:
    los.sort(key=lambda x: -x[2])
    acc = los[0][-1]
    print(acc)

