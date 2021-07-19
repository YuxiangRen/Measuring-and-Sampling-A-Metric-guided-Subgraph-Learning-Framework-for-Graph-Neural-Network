from __future__ import division
from __future__ import print_function
import pickle
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter

from utils_citeseer_pubmed import load_data, accuracy
from models import GCN, SpGAT, Linear

# Training settings
# torch.cuda.set_device(2)
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--test_format', type=str, default="concat", help='foramt of test: vote, concat, or other')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
AAAA, features, labels, idx_train, idx_val, idx_test = load_data('cora')

model = GCN(nfeat=features.shape[1], nhid=args.hidden, 
            nclass=int(labels.max()) + 1, dropout=args.dropout, alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    #adj = adj.cuda()
    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()

features, labels = Variable(features), Variable(labels)

def train(epoch, features, adj, labels, idx_train, idx_val):
    t = time.time()
    if args.cuda:
        adj = adj.cuda()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch), 'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()), 'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()), 'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), loss_train.data.item(), acc_train.data.item(), acc_val.data.item()




def test(content_g, features, labels):
    final_train_hidden, final_test_hidden, final_train_divide, final_test_divide = {}, {}, {}, {}
    for i in idx_test:
        final_test_hidden[i.data.item()] = 0
        final_test_divide[i.data.item()] = 0
    for i in idx_train:
        final_train_hidden[i.data.item()] = 0
        final_train_divide[i.data.item()] = 0

    with torch.no_grad():
        model.eval()
        for subgraph in content_g:
            index = content_g[subgraph]['index_subgraph']
            idx_train_subgraph, idx_test_subgraph = content_g[subgraph]['idx_train'], content_g[subgraph]['idx_test']
            if(len(idx_test_subgraph) == 0):
                continue
            true_train_idx = content_g[subgraph]['idx_train_list']
            true_test_idx = content_g[subgraph]['idx_test_list']
            adj = content_g[subgraph]['adj']
            adj = torch.FloatTensor(np.array(adj.todense()))
            if args.cuda:
                adj = adj.cuda()
            output = model(features[index], adj, concat=True)
            acc_test = accuracy(output[idx_test_subgraph], labels[index][idx_test_subgraph])

            for i in range(len(true_test_idx)):
                final_test_hidden[int(true_test_idx[i])] += output[idx_test_subgraph][i]
                final_test_divide[int(true_test_idx[i])] += 1
            for i in range(len(true_train_idx)):
                final_train_hidden[int(true_train_idx[i])] += output[idx_train_subgraph][i]
                final_train_divide[int(true_train_idx[i])] += 1
    
    train_feat_cla, train_gt_cla = torch.LongTensor([]), []
    for i in final_train_divide:
        if(final_train_divide[i] > 0):
            final_train_hidden[i] /= final_train_divide[i]
            train_feat_cla = torch.cat([train_feat_cla, final_train_hidden[i]])
            train_gt_cla.append(labels[i])

    train_feat_cla = train_feat_cla.reshape(-1, args.hidden)
    train_gt_cla = torch.LongTensor(train_gt_cla)

    no_test_list = []
    for i in final_test_hidden:
        if(final_test_hidden[i] is 0):
            no_test_list.append(i)

    #resample subgraphs for the test sample without corresponding subgraphs
    if(len(no_test_list) > 0):
        subgraph_set_dict, subgraph_size = subgraph_sample(init_idx_list = no_test_list, kstep=4, ratio=0.2)
        with torch.no_grad():
            model.eval()
            for i in no_test_list:
                index = subgraph_set_dict[i]['index_subgraph']
                adj = subgraph_set_dict[i]['adj']
                adj = torch.FloatTensor(np.array(adj.todense()))
                if args.cuda:
                    adj = adj.cuda()
                output = model(features[index], adj, concat=True)
                final_test_hidden[i] += output[0]
                final_test_divide[i] += 1 

    test_feat_cla, test_gt_cla = torch.LongTensor([]), []
    for i in final_test_hidden:
        if(final_test_divide[i] > 0):
            # acc_num += 1
            final_test_hidden[i] /= final_test_divide[i]
            test_feat_cla = torch.cat([test_feat_cla, final_test_hidden[i]])
            test_gt_cla.append(labels[i])


    test_feat_cla = test_feat_cla.reshape(-1, args.hidden)
    test_gt_cla = torch.LongTensor(test_gt_cla)

    model_cla = Linear(nfeat=args.hidden, nclass=int(labels.max()) + 1, dropout=0.3)
    model_cla.state_dict()['linear1.W'] = model.state_dict()['conv2.W']
    if args.cuda:
        model_cla.cuda()
        train_feat_cla = train_feat_cla.cuda()
        train_gt_cla = train_gt_cla.cuda()
        test_feat_cla = test_feat_cla.cuda()
        test_gt_cla = test_gt_cla.cuda()
    
    optimizer_cla = optim.Adam(model_cla.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_test_cla = []
    for iter in range(1000):
        model_cla.train()
        optimizer_cla.zero_grad()
        output_train_cla = model_cla(train_feat_cla)
        loss_train_cla = F.nll_loss(output_train_cla, train_gt_cla)
        acc_train = accuracy(output_train_cla, train_gt_cla)
        loss_train_cla.backward()
        optimizer_cla.step()
        model_cla.eval()
        output_test_cla = model_cla(test_feat_cla)
        acc_test = accuracy(output_test_cla, test_gt_cla)
        acc_test_cla.append(acc_test)

    
    model_cla.eval()
    output_test_cla = model_cla(test_feat_cla)
    acc_test = accuracy(output_test_cla, test_gt_cla)


    print("Concat format Test set results:", max(acc_test_cla))

    return acc_test

## Train model
t_total, loss_values, bad_counter, best, best_epoch = time.time(), [], 0, args.epochs + 1, 1

#load ripple walk subgraphs
from subgraph_sample_cora import subgraph_sample
# destination_folder_path = './sampled_subgraph/'
# destination_file_name_g = 'cora_metricguide_smoothnessration5_num50_3'
# g = open(destination_folder_path + destination_file_name_g, 'rb')
# content_g = pickle.load(g)
# g.close()
content_g, subgraph_size_load = subgraph_sample(number_subgraph=50, kstep=5, ratio=0.2, pn=0.5)

train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_acc_list, iter = [], [], [], [], [], 0

for epoch in range(args.epochs):
    for subgraph in content_g:
        iter += 1
        index = content_g[subgraph]['index_subgraph']
        if(len(content_g[subgraph]['idx_train'])*len(content_g[subgraph]['idx_val']) == 0):
            continue
        adj = content_g[subgraph]['adj']
        adj = torch.FloatTensor(np.array(adj.todense()))
        val_loss, train_loss, train_acc, val_acc = train(iter,features[index],adj,labels[index], 
                content_g[subgraph]['idx_train'], content_g[subgraph]['idx_val'])
        acc_test = 0#test(content_g, features, labels)
        
        loss_values.append(val_loss)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        test_acc_list.append(acc_test)

        torch.save(model.state_dict(), '{}.corapkl'.format(iter))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = iter
            bad_counter = 0
        else:
            bad_counter += 1

        # if bad_counter == args.patience:
        #     break

        files = glob.glob('*.corapkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

files = glob.glob('*.corapkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print('The max test acc', max(test_acc_list))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.corapkl'.format(best_epoch)))


# Testing with different formats: vote, concat, or others
if(args.test_format == "vote"):
    final_test_preds, final_test_results = {},[]
    for i in idx_test:
        final_test_preds[i.data.item()] = []

    with torch.no_grad():
        model.eval()
        for subgraph in content_g:
            index = content_g[subgraph]['index_subgraph']
            idx_test_subgraph = content_g[subgraph]['idx_test']
            if(len(idx_test_subgraph) == 0):
                continue
            true_test_idx = content_g[subgraph]['idx_test_list']
            # print('len test idx', max(true_test_idx))
            adj = content_g[subgraph]['adj']
            adj = torch.FloatTensor(np.array(adj.todense()))
            if args.cuda:
                adj = adj.cuda()
            output = model(features[index], adj)

            temp_preds = output[idx_test_subgraph].max(1)[1].type_as(labels)
            assert(temp_preds.size()[0] == len(true_test_idx))

            for i in range(len(true_test_idx)):
                final_test_preds[int(true_test_idx[i])].append(temp_preds[i])

    no_test_list = []
    for i in final_test_preds:
        if(len(final_test_preds[i]) == 0):
            no_test_list.append(i)
 
    # resample subgraphs for the test sample without corresponding subgraphs
    if(len(no_test_list) > 0):
        subgraph_set_dict, subgraph_size = subgraph_sample(init_idx_list = no_test_list)
        with torch.no_grad():
            model.eval()
            for i in no_test_list:
                index = subgraph_set_dict[i]['index_subgraph']
                adj = subgraph_set_dict[i]['adj']
                adj = torch.FloatTensor(np.array(adj.todense()))
                if args.cuda:
                    adj = adj.cuda()
                output = model(features[index], adj)
                # print('output size', output.max(1)[1].type_as(labels)[0])
                final_test_preds[i].append(output.max(1)[1].type_as(labels)[0])
            
    
    for idx in final_test_preds:
        final_test_results.append(Counter(final_test_preds[idx]).most_common(1)[0][0].data.item())

    final_test_results = torch.LongTensor(final_test_results)
    correct = final_test_results.eq(labels[idx_test]).double()
    correct = correct.sum()
    acc_test = correct / len(labels[idx_test])

    print("Vote format Test set results:", "acc= {:.4f}".format(acc_test))#data[0])),

elif(args.test_format == "concat"):
    acc_test = test(content_g, features, labels)
    print("Concat format Test set results:", acc_test)



data = {'train_loss': train_loss_list, 'train_acc': train_acc_list,
        'val_loss': val_loss_list, 'val_acc': val_acc_list, 'test_acc': acc_test}

