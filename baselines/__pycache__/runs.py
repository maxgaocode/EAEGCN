import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from utils import accuracy, preprocess_data
from model import GCN, MAGCN

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='squirrel')# 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel'   film
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=900, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--gcn', type=int, default=0, help='use the gcn model')
args = parser.parse_args()

device = 'cuda'
# parser.add_argument('--seed', type=int, default=42, help='Patience')
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

def gae_for(g, features, nclass):

    net = MAGCN(g, features.size()[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num).cuda()



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

        if args.gcn == 0:
            logp = net(features)


        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_acc = accuracy(logp[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        if args.gcn == 0:
            logp = net(features)

        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
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
        if epoch %100 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))


    if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
        los.sort(key=lambda x: x[1])
        acc = los[0][-1]
        print('testacc')
        print(acc)
    else:
        los.sort(key=lambda x: -x[2])
        acc = los[0][-1]
        print('testacc')
        print(acc)


    return acc
if __name__ == '__main__':
    train_ratio=0.1
    acclist = []
    for i in range(6):
        train_ratio += 0.1
        g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, train_ratio)
        print(train_ratio)
        features = features.to(device)
        labels = labels.to(device)
        rain = train.to(device)
        test = test.to(device)
        val = val.to(device)
        g = g.to(device)
        deg = g.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm

        acc=gae_for(g, features, nclass)
        acclist.append(acc)
    print(acclist)


####GCN
#cornell  [0.5405405405405406, 0.5405405405405406, 0.5405405405405406, 0.5135135135135135, 0.5135135135135135, 0.5675675675675675]
#texas  [0.6756756756756757, 0.6756756756756757, 0.7027027027027027, 0.7567567567567568, 0.7567567567567568, 0.7567567567567568]
#wisconsin  [0.7058823529411765, 0.7254901960784313, 0.6666666666666666, 0.7254901960784313, 0.7450980392156863, 0.7254901960784313]


#film  [0.2730263157894737, 0.29276315789473684, 0.28092105263157896, 0.2743421052631579, 0.2710526315789474, 0.2361842105263158]
# squirrel  [0.4159462055715658, 0.4562920268972142, 0.4918347742555235, 0.48895292987512007, 0.4812680115273775, 0.5100864553314121]
#chameleon [0.5460526315789473, 0.5942982456140351, 0.625, 0.6359649122807017, 0.6403508771929824, 0.6403508771929824]


####MIXHOP




#relu+Remove
#cornell  [0.7297297297297297, 0.7027027027027027, 0.7297297297297297, 0.7837837837837838, 0.8378378378378378, 0.7027027027027027]
#texas 
#wisconsin [0.803921568627451, 0.8431372549019608, 0.8627450980392157, 0.9215686274509803, 0.9215686274509803, 0.9019607843137255]

#chameleon  [0.5460526315789473, 0.5153508771929824, 0.5899122807017544, 0.506578947368421, 0.6096491228070176, 0.618421052631579]
#squirrel [0.32372718539865514, 0.345821325648415, 0.3371757925072046, 0.3573487031700288, 0.37752161383285304, 0.4322766570605187]



###tanh+Remove
#chameleon [0.5482456140350878, 0.5921052631578947, 0.6008771929824561, 0.6206140350877193, 0.6293859649122807, 0.6359649122807017]
#cornell    [0.5405405405405406, 0.5135135135135135, 0.6486486486486487, 0.5945945945945946, 0.6486486486486487, 0.6756756756756757]


