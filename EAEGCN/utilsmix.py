

import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *
import numpy as np


import matplotlib.pyplot as plt
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

'''
def drawnetworklabel(G, G1, cos, labels, drawg=0):
    withnodeID = 0

    srcloca, dstloca = torch.where(cos > 0)
    listlables = labels.tolist()
    d = 0
    for s, q in zip(srcloca, dstloca):

        labelsrc = listlables[s]
        labeldst = listlables[q]
        if labelsrc == labeldst:
            d += 1

    for i in range(len(dstloca)):
        G1.add_edge(int(srcloca[i]), int(dstloca[i]))

    return G1
'''

def count_labelconsistent(edge, labels, both=True):
    if both:
        # U = [e[0] for e in edge]
        # V = [e[1] for e in edge]

        c1 = 0
        c2 = 0
        lab = labels.tolist()
        for e in edge:
            if lab[e[0]] == lab[e[1]]:
                c1 += 1
            else:
                c2 += 1

    else:
        adj = edge  # 这里还需要做一个转换
        src, dst = torch.where(adj > 0)

        listlables = labels.tolist()
        c1 = 0
        for s, q in zip(src, dst):

            labelsrc = listlables[s]
            labeldst = listlables[q]
            if labelsrc == labeldst:
                c1 += 1
        print(c1)
    return c1 / len(src)





import numpy as np
import sys
import pickle as pkl

def preprocess_data(dataset, train_ratio):

    if dataset in ['cora', 'citeseer', 'pubmed']:

        edge = np.loadtxt('../homophily/{}.edge'.format(dataset), dtype=int).tolist()
        feat = np.loadtxt('../homophily/{}.feature'.format(dataset))
        labels = np.loadtxt('../homophily/{}.label'.format(dataset), dtype=int)
        train = np.loadtxt('../homophily/{}.train'.format(dataset), dtype=int)
        val = np.loadtxt('../homophily/{}.val'.format(dataset), dtype=int)
        test = np.loadtxt('../homophily/{}.test'.format(dataset), dtype=int)

        names = ['ty', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../homophily/ind.{}.{}".format(dataset, 'graph'), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        _, graph = tuple(objects)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = torch.FloatTensor(adj.to_dense())

        nclass = len(set(labels.tolist()))
        print(dataset, nclass)

        feat = normalize_features(feat)
        feat = torch.FloatTensor(feat)


        feasim = torch.mm(feat, feat.t())
        feasim = torch.add(torch.softmax(feasim, dim=1), adj)

        cos = torch.sign(torch.sign(feasim - torch.mean(feasim, dim=1)) + 1)
        cos = torch.mul(cos, adj)

        g = nx.from_numpy_matrix(cos.numpy())
        g = DGLGraph(g)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)



        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, nclass, feat, labels, train, val, test
   

    # datasets with 5 class
    elif dataset in ['film','cornell', 'texas', 'wisconsin']:

        graph_adjacency_list_file_path = '../heterophily/{}/out1_graph_edges.txt'.format(dataset)
        graph_node_features_and_labels_file_path = '../heterophily/{}/out1_node_feature_label.txt'.format(dataset)

        G = nx.DiGraph()
        G1 = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        if dataset == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint16)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                    G1.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                    G1.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        ##row, col = np.where(adj.todense() > 0)
        if dataset == 'film':
            row, col = np.where(adj.todense() > 0)

            U = row.tolist()
            V = col.tolist()
            g = dgl.graph((U, V))
            features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])],
                                dtype=float)
            labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])], dtype=int)
        else:
            g = DGLGraph(adj)

            features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])


        n = len(labels.tolist())

        idx = [i for i in range(n)]
        # random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])


        features = torch.FloatTensor(features)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = torch.FloatTensor(adj.to_dense())

        bela = 0.9
        feasim = torch.mm(features, features.t())
        feasim = torch.add(bela * torch.softmax(feasim, dim=1), (1-bela) * adj)
        cos = torch.sign(torch.sign(feasim - torch.mean(feasim, dim=1)) + 1)


        g = nx.from_numpy_matrix(cos.numpy())
        g = DGLGraph(g)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g) #dgl.add_self_loop(g)
        


        nclass = len(set(labels.tolist()))

        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)
