# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:48:18 2023

@author: Pandadada
"""


import numpy as np
import os
import sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath(
'E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_0.0'))

from graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
import random
import ot
import networkx as nx
import matplotlib.pyplot as plt
import copy
from ot_distances import Fused_Gromov_Wasserstein_distance
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
# from data_loader import load_local_data,histog,build_noisy_circular_graph
# from FGW import init_matrix,gwloss  # lib 0.0 no need
# from FGW import cal_L,tensor_matrix,gwloss
import scipy.stats as st

#%% 
G1=Graph() # target 
# G1.add_attributes({0:'guardians',1:'groot',2:'star',3:'guardians2'})    # add features to nodes
G1.add_attributes({0:0, 1:7, 2:3, 3:1})    # add features to nodes

G1.add_edge((0,1))
G1.add_edge((0,2))
G1.add_edge((0,3))

G2_nodummy=Graph()
# G2_nodummy.add_attributes({0:'guardians',1:'groot',2:'star'})    # add features to nodes
G2_nodummy.add_attributes({0:0, 1:7, 2:3})    # add features to nodes

G2_nodummy.add_edge((0,1))
G2_nodummy.add_edge((0,2))

# %%
# G2_nodummy = copy.deepcopy(G11)
# G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)
G2 = copy.deepcopy(G2_nodummy)
G2.add_attributes({len(G2.nodes()): 0})  # add dummy

# %%  The followings are fixed
g1 = G1.nx_graph
g2 = G2.nx_graph
g2_nodummy = G2_nodummy.nx_graph

# %% check if every pair of nodes have path
# n1 = len(G1.nodes())
# try:
#     for ii in range(n1):
#           nx.shortest_path_length(g1,source=0,target=ii)
# except:
#     print("oops2")
#     continue

# %%
vmin = 0
vmax = 9  # the range of color

# plt.figure(figsize=(8, 5))
# create some bugs in the nx.draw_networkx, don't know why.
# draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
# draw_rel(g2, vmin=vmin, vmax=vmax,
#          with_labels=True, shiftx=3, draw=False)
# plt.title('Two graphs. Color indicates the label')
# plt.show()

# %% weights and feature metric
p1 = ot.unif(len(G1.nodes()))
# ACTUALLY NOT USED IN THE ALGORITHM
p2_nodummy = 1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])
p2 = np.append(p2_nodummy, 1-sum(p2_nodummy))

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
str_metric = 'shortest_path'
# str_metric = 'adj'

# %% use the function from FGWD all the time
thresh = 0.004
# WD
# dw, log_WD, transp_WD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
#     alpha=0, features_metric=fea_metric, method='shortest_path', loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy)
# fig=plt.figure(figsize=(10,8))
# plt.title('WD coupling')
# draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()

# GWD
# dgw, log_GWD, transp_GWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
#     alpha=1, features_metric=fea_metric, method='shortest_path', loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy)
# fig=plt.figure(figsize=(10,8))
# plt.title('GWD coupling')
# draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()

# FGWD
alpha = 0.5
dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy)

fig = plt.figure(figsize=(10, 8))
plt.title('FGWD coupling')
draw_transp(G1, G2, transp_FGWD, shiftx=2, shifty=0.5, thresh=thresh,
    swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
plt.show()

# %% FGWD, find alpha
# alld=[]
# x=np.linspace(0,1,10)
# for alpha in x:
#     d,log,transp=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
#     alld.append(d)
# fig=plt.figure(figsize=(10,8))
# plt.plot(x,alld)
# plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(x[np.argmax(alld)]))
# plt.xlabel('Alpha')
# plt.xlabel('FGW dist')
# plt.show()

# # optimal matching
# fig=plt.figure(figsize=(10,8))
# thresh=0.004
# alpha_opt=x [ alld.index(max(alld)) ]
# dfgw_opt,log_FGWD_opt,transp_FGWD_opt=Fused_Gromov_Wasserstein_distance(alpha=alpha_opt,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
# # d=dfgw.graph_d(g1,g2)
# # plt.title('FGW coupling, dist : '+str(np.round(dfgw,3)),fontsize=15)
# plt.title('FGW coupling, alpha = opt')
# draw_transp(G1,G2,transp_FGWD_opt,shiftx=2,shifty=0.5,thresh=thresh,
#             swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()

# print('Wasserstein distance={}, Gromov distance={} \nFused Gromov-Wasserstein distance for alpha {} = {}'.format(dw,dgw,alpha,dfgw))

#%% NeMa
alpha = 0.5
dfgw, log_NeMa, transp_NeMa, M, C1, C2 = NeMa( alpha=alpha, features_metric=fea_metric, method=str_metric, G1, G2 )

fig = plt.figure(figsize=(10, 8))
plt.title('FGWD coupling')
draw_transp(G1, G2, transp_FGWD, shiftx=2, shifty=0.5, thresh=thresh,
    swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
plt.show()

# %%
thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-4

# dfgw=dfgw/N # modified obj values 
# DFGW[num] = dfgw
# if dfgw < thre1:
#     yes1 += 1
# if dfgw < thre2:
#     yes2 += 1

# num += 1
# print(num)

# # %% check the features and structure
# if Is_info:
#     index = np.argwhere(transp_FGWD[:, 0:-1] > 1e-3)
#     # Get the indices that would sort the second column in ascending order
#     sort_indices = np.argsort(index[:, 1])
#     index = index[sort_indices]
#     # feature
#     Features_source = list(g1._node.values())
#     print("Features of subgraph within the source graph:")
#     for source in index[:, 0]:  # source is int
#         print(Features_source[source])

#     print("Features of the query graph:")
#     Features_target = list(g2_nodummy._node.values())
#     for target in index[:, 1]:
#         print(Features_target[target])

#     # structure
#     print("Neighbours of source subgraph:")
#     Structure_keys = list(g1._node.keys())
#     Structure_source = list(g1._adj.values())
#     Structure_source2 = {}  # the subgraph within the large graph, but with irrelevant nodes
#     for source in index[:, 0]:
#         Structure_source2[Structure_keys[source]
#                           ] = Structure_source[source]

#     temp_keys = list(Structure_source2.keys())
#     for key in temp_keys:
#         for k in Structure_source2[key].copy():
#             if k not in temp_keys:
#                 # delete the irrelevant nodes
#                 Structure_source2[key].pop(k, None)
#         print(Structure_source2[key])

#     print("Neighbours of query graph:")
#     Structure_target = list(g2_nodummy._adj.values())
#     for target in index[:, 1]:
#         print(Structure_target[target])

#     # Adj matrix

#     def generate_adjacency_matrix(graph_dict):
#         # Get all unique nodes from the dictionary keys
#         nodes = list(graph_dict.keys())
#         num_nodes = len(nodes)

#         # Initialize an empty adjacency matrix with zeros
#         adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

#         # Iterate over the graph dictionary
#         for node, connections in graph_dict.items():
#             # Get the index of the current node
#             node_index = nodes.index(node)

#             # Iterate over the connected nodes
#             for connected_node in connections.keys():
#                 # Get the index of the connected node
#                 connected_node_index = nodes.index(connected_node)

#                 # Set the corresponding entry in the adjacency matrix to 1
#                 adjacency_matrix[node_index][connected_node_index] = 1

#         return adjacency_matrix

#     adjacency_subgraph = generate_adjacency_matrix(Structure_source2)
#     print("Adjacency matrix within the source graph")
#     print(adjacency_subgraph)

#     adjacency_query = generate_adjacency_matrix(g2_nodummy._adj)
#     print("Adjacency matrix of query graph")
#     print(adjacency_query)

# # %%
# print('Rate 1:',yes1/Num)
# print('Rate 2:',yes2/Num)
# print('STD:',np.std(DFGW))

# DFGW_set.append(DFGW)
# Percent1.append(yes1/Num)
# Percent2.append(yes2/Num)

# Mean.append(np.mean(DFGW))
# STD.append(np.std(DFGW))

# #create 95% confidence interval for population mean weight
# lower, upper = st.norm.interval(confidence=0.95, loc=np.mean(DFGW), scale=st.sem(DFGW))
# Lower.append(lower)
# Upper.append(upper)

# # %% boxplot
# fig, ax = plt.subplots()
# # ax.set_title('Hide Outlier Points')
# ax.boxplot(DFGW_set, showfliers=False, showmeans=False)
# # %% plot mean and STD
# plt.figure()
# plt.plot(np.array(Pw), np.array(Mean), 'k-+')
# # plt.fill_between(np.array(Pw), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD), alpha=0.5) # alpha here is transparency
# plt.fill_between(np.array(Pw), np.array(Lower), np.array(Upper), facecolor = 'k',alpha=0.5) # alpha here is transparency
# plt.grid()
# # plt.xlabel('Size of test graph')
# # plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.ylabel('Mean and 95% confidence interval')
# # %% plot percentage
# plt.figure()
# plt.plot(np.array(Pw), np.array(Percent1),'k-x', label='exact match')
# plt.plot(np.array(Pw), np.array(Percent2),'k--.', label='approx match')
# plt.grid()
# # plt.xlabel('Size of test graph')
# # plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.ylabel('Success rate')
# plt.legend()
#%% subsitute back the transport matrix
# n1 = len(G1.nodes())
# n2 = len(G2.nodes())
# constC,hC1,hC2=init_matrix(C1,
#                             C2[0:n2-1,0:n2-1],
#                             transp_FGWD[:,0:len(transp_FGWD[0])-1],
#                             p1,
#                             p2_nodummy,
#                             loss_fun='square_loss')
# check_gwloss=gwloss(constC,hC1,hC2,transp_FGWD)
# print(check_gwloss)
# check_wloss=np.sum(transp_FGWD*M)
# print(check_wloss)
# check_fgwloss = (1-alpha)*check_wloss+alpha*check_gwloss
# print(check_fgwloss)
# %% subsitute back the transport matrix
# n1 = len(G1.nodes())
# n2 = len(G2.nodes())
# # constC,hC1,hC2=init_matrix(C1,
# #                             C2[0:n2-1,0:n2-1],
# #                             transp_FGWD[:,0:len(transp_FGWD[0])-1],
# #                             p1,
# #                             p2_nodummy,
# #                             loss_fun='square_loss')
# # check_gwloss=gwloss(constC,hC1,hC2,transp_FGWD)
# check_gwloss=gwloss(cal_L(C1,C2),transp_FGWD)
# print(check_gwloss)
# check_wloss=np.sum(transp_FGWD*M)
# print(check_wloss)
# check_fgwloss = (1-alpha)*check_wloss+alpha*check_gwloss
# print(check_fgwloss)
