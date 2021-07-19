import numpy as np
import random
import scipy.sparse as sp
import sys
import torch
import pickle
import networkx as nx

from utils_citeseer_pubmed import load_data, normalize_adj


def load_graph(dataset_str = 'pubmed'):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    return graph


def to_add(features, poten_list, potential, smoothness, ratio=0.2):
    if(len(poten_list) == 0):
        return True
    else:
        what_to_reture = True
        for idx in poten_list:
            if(torch.sum((features[idx]-features[potential])*(features[idx]-features[potential])) < (smoothness*ratio)):
                what_to_reture = False
                break
    return what_to_reture


def subgraph_sample(init_idx_list=None, number_subgraph=50, kstep=3, ratio=0.2, pn=0.5):
    # load graphs and related data
    adj, features, labels, idx_train, idx_val, idx_test = load_data('pubmed')
    graph = load_graph()

    # compute the smoothness of the target graph
    num_edge, smoothness = 0, 0
    for i in range(len(graph)):
        num_edge += len(graph[i])
        for j in graph[i]:
            smoothness += (features[i] - features[j])*(features[i] - features[j])

    smoothness = torch.sum(smoothness)/num_edge
    print('the smoothness', smoothness)
    print('\n')

    num_train, num_val, num_test = idx_train.size()[0], idx_val.size()[0], idx_test.size()[0]

    subgraph_set_dict, subgraph_size  = {}, []
    if(init_idx_list is None):

        index_subgraph, itera = [], 0
        while(itera < number_subgraph):
            # select initial node
            if(len(index_subgraph)<2001):
                index_subgraph = list(set(index_subgraph + [np.random.randint(len(graph))]))#[np.random.randint(620, 1620)]))
            else:
                index_subgraph = [np.random.randint(len(graph))]
            for step in range(kstep):
                neighbors = []
                #select part of the neighbor nodes and insert them into the current subgraph
                for i in index_subgraph:
                    poten_list, poten_distance_list, poten_distance_list_tosort = [], [], []
                    for potential in graph[i]:
                        if((potential not in index_subgraph) & to_add(features, [i], potential, smoothness, ratio=ratio)):
                            poten_distance_list.append(torch.sum((features[potential]-features[i])*(features[potential]-features[i])))
                            poten_distance_list_tosort.append(torch.sum((features[potential]-features[i])*(features[potential]-features[i])))
                            poten_list.append(potential)

                    # if(len(poten_list) < 2):
                    #     neighbors = neighbors + poten_list
                    # else:
                    #     poten_distance_list_tosort.sort(reverse=True)
                    #     for z in range(int(min(len(poten_list)*pn + 1, len(poten_list)))):
                    #         for y in range(len(poten_list)):
                    #             if(poten_distance_list_tosort[z] == poten_distance_list[y]):
                    #                 neighbors.append(poten_list[y])

                    index_subgraph = index_subgraph + poten_list#neighbors
                index_subgraph = list(set(index_subgraph))
                if(len(index_subgraph)>2000):
                    break

            idx_train_subgraph, idx_val_subgraph, idx_test_subgraph, idx_train_list, idx_test_list = [], [], [], [], []

            index_subgraph = list(set(index_subgraph + list(np.array(idx_val))))
            for i in range(len(index_subgraph)):
                if(index_subgraph[i] in idx_train):
                    idx_train_subgraph.append(i)
                    idx_train_list.append(index_subgraph[i])
                elif(index_subgraph[i] in idx_val):
                    idx_val_subgraph.append(i)
                elif(index_subgraph[i] in idx_test):
                    idx_test_subgraph.append(i)
                    idx_test_list.append(index_subgraph[i])


            G = nx.from_dict_of_lists(graph)
            g = G.subgraph(index_subgraph)
            adj =nx.adjacency_matrix(g)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            
            # if(len(index_subgraph)>1500):
            print(itera + 1, 'th subgraph saved, the size is', len(index_subgraph))
            subgraph_size.append(len(index_subgraph))

            subgraph_set_dict[itera] = {'index_subgraph': torch.LongTensor(index_subgraph), 'adj':adj, 'idx_train': torch.LongTensor(idx_train_subgraph), 'idx_val': torch.LongTensor(idx_val_subgraph),
                    'idx_test': torch.LongTensor(idx_test_subgraph), 'idx_train_list': idx_train_list, 'idx_test_list':idx_test_list}
            itera += 1
   
    else:
        print('Sampling from specified initial test nodes.\n')
        for init_idx in init_idx_list:
            index_subgraph = [init_idx]
            for step in range(kstep):
                neighbors = []
                #select part of the neighbor nodes and insert them into the current subgraph
                for i in index_subgraph:
                    poten_list, poten_distance_list, poten_distance_list_tosort = [], [], []
                    for potential in graph[i]:
                        if((potential not in index_subgraph) & to_add(features, [i], potential, smoothness, ratio=ratio)):
                            poten_distance_list.append(torch.sum((features[potential]-features[i])*(features[potential]-features[i])))
                            poten_distance_list_tosort.append(torch.sum((features[potential]-features[i])*(features[potential]-features[i])))
                            poten_list.append(potential)

                    # if(len(poten_list) < 2):
                    #         neighbors = neighbors + poten_list
                    # else:
                    #     poten_distance_list_tosort.sort(reverse=True)
                    #     for z in range(int(min(len(poten_list)*pn + 1, len(poten_list)))):
                    #         for y in range(len(poten_list)):
                    #             if(poten_distance_list_tosort[z] == poten_distance_list[y]):
                    #                 neighbors.append(poten_list[y])
                                    
                    index_subgraph = index_subgraph + poten_list#neighbors
                index_subgraph = list(set(index_subgraph))
            
            G = nx.from_dict_of_lists(graph)
            g = G.subgraph(index_subgraph)
            adj =nx.adjacency_matrix(g)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))

            index_subgraph = torch.LongTensor(index_subgraph)
            # print('resample subgraph size is', len(index_subgraph))
            subgraph_size.append(len(index_subgraph))
            subgraph_set_dict[init_idx] = {'index_subgraph': index_subgraph, 'adj':adj}
        
    return subgraph_set_dict, subgraph_size



if __name__ == "__main__":

    sampled_subgraph_dict,subgraph_size = subgraph_sample(init_idx_list=None,number_subgraph=50,kstep=7,ratio=0.2)
    print('avg subgraph size', sum(subgraph_size)/len(subgraph_size))
    print('num graph', len(subgraph_size))
    f = open('./sampled_subgraph/pubmed_mg_kstep4_sr02', 'wb')
    pickle.dump({0: sampled_subgraph_dict, 1: subgraph_size}, f)
    f.close()
