import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolutionLayer, SpGraphAttentionLayer, LinearLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv1 = GraphConvolutionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, not_final=True)
        
        self.add_module('conv1', self.conv1)

        self.conv2 = GraphConvolutionLayer(nhid, nclass, dropout=dropout, alpha=alpha, not_final=False)
        
    def forward(self, x, adj, concat=False):
        x = F.dropout(x, self.dropout, training=self.training)#dropout掉输入数据层
        x = self.conv1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)#dropout掉隐层
        if(concat):
            x = self.conv2(x, adj, concat=concat)
            return x
        else:
            x = self.conv2(x, adj)
            return F.log_softmax(x, dim=1)

class Linear(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(Linear, self).__init__()
        self.dropout = dropout

        self.conv1 = LinearLayer(nfeat, nclass, dropout=dropout)
        self.add_module('linear1', self.conv1)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)#dropout掉输入数据层
        x = self.conv1(x)
        return F.log_softmax(x, dim=1)

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

