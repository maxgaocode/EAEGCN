import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl import function as fn
import torch.nn.functional as F

import math
def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


class MALayer(nn.Module):
    def __init__(self, g, in_dim, dropedge, Remove, **kwargs):
        super(MALayer, self).__init__()
        self.g = g
        self.dropedge = nn.Dropout(dropedge)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)




    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()   ##the edge weight
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropedge(e)

        return {'e': e}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']



class MAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, dropedge, eps, layer_num, Remove=False, **kwargs):
        super(MAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.dropedge = dropedge

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(MALayer(self.g, hidden_dim, dropedge, Remove, **kwargs))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)



from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes,dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.dropout = dropout
    def forward(self, g, inputs):
        h = F.dropout(inputs, p=self.dropout, training=self.training)
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return  F.log_softmax(h, 1)






