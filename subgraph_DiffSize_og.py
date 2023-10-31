# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:24:22 2023

@author: Pandadada
"""
import numpy as np
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# sys.path.append(os.path.realpath('../lib'))
# sys.path.append(os.path.realpath(‘E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_0.0'))

from lib1.graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
import random
import ot
import networkx as nx
import matplotlib.pyplot as plt
import copy
from lib1.ot_distances import Fused_Gromov_Wasserstein_distance
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
# from data_loader import load_local_data,histog,build_noisy_circular_graph
# from FGW import init_matrix,gwloss  # lib 0.0 no need
# from FGW import cal_L,tensor_matrix,gwloss
import scipy.stats as st
import time


N = 8 # size of query
S = 6 # size of subgraph
N3 = 30
deg = 5
numfea = 15

def build_G1(G, N2=30, numfea=3, pw=0.5):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    # Fea = np.linspace(0,20,numfea)
    Fea = list(range(0, numfea))

    L = len(G.nodes())
    # G.add_nodes(list(range(N2)))

    NN = N2+L  # total number of nodes in test graph
    for i in range(L, NN):
        # G.add_one_attribute(i,v[i-L])
        fea = random.choice(Fea)
        G.add_one_attribute(i, fea)
    for i in range(NN):
        for j in range(i+1, NN):
            if j != i and j not in range(L):  # no additional edge within the subgraph
            # if j != i and j not in range(L) and j not in G.nx_graph._adj[i].keys():
                r = np.random.rand()  # uniform betweeen [0,1)
                if r < pw:
                    G.add_edge((i, j))

    return G

def build_fully_graph(N=30, numfea=3):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    g = Graph()
    g.add_nodes(list(range(N)))
    # Fea = np.linspace(0,20,numfea)
    Fea = list(range(0, numfea))
    for i in range(N):
        # g.add_one_attribute(i,v[i])
        # g.add_one_attribute(i,2)
        fea = random.choice(Fea)
        g.add_one_attribute(i, fea)
        for j in range(i+1, N):
            if j != i:
                g.add_edge((i, j))

    return g

#%% query 
# G11=Graph() # community 1
# G11.add_attributes({0:0,1:1,2:2,3:3,4:4,5:4,6:4})    # add color to nodes
# G11.add_edge((0,1))
# G11.add_edge((0,2))
# G11.add_edge((0,3))
# G11.add_edge((1,2))
# G11.add_edge((1,3))
# G11.add_edge((2,3))

# G11.add_edge((3,4))
# G11.add_edge((4,5))
# G11.add_edge((4,6))
# G11.add_edge((5,6))

#%% subgraph
# G12=Graph() # community 1
# G12.add_attributes({0:0,1:1,2:2,3:3,4:4,5:4,6:4,7:4})    # add color to nodes
# G12.add_edge((0,1))
# G12.add_edge((0,2))
# G12.add_edge((0,3))
# G12.add_edge((1,2))
# G12.add_edge((1,3))
# G12.add_edge((2,3))

# G12.add_edge((3,4))
# G12.add_edge((4,5))
# G12.add_edge((4,6))
# G12.add_edge((4,7))
# G12.add_edge((5,6))
# G12.add_edge((5,7))
# G12.add_edge((6,7))

#%% query
G11 = build_fully_graph(N=N, numfea=2)
g11 = G11.nx_graph
    
#%% subgraph
G12 = build_fully_graph(N=S, numfea=2)
g12 = G12.nx_graph

#%% test
G13 = copy.deepcopy(G12)  # initialize with subgraph
# G111=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
# G112=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
# G1 = Graph(merge_graph(G111.nx_graph,G112.nx_graph))
N2 = N3 - S
# pw2=pw1
# pw2 = Pw[ NN3.index(N3) ]
pw2 = deg / (N3-1)
G1 = build_G1(G13, N2=N2, numfea=numfea, pw=pw2)

#%%
G2_nodummy = copy.deepcopy(G11)
# G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)        
g1 = G1.nx_graph
g2_nodummy = G2_nodummy.nx_graph

Large = 1e6
G2 = copy.deepcopy(G2_nodummy)
G2.add_attributes({len(G2.nodes()): 0})  # add dummy   

#%% 
vmin = 0
vmax = 9  # the range of color

plt.figure()
draw_rel(g2_nodummy, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
draw_rel(g12, vmin=vmin, vmax=vmax, with_labels=True, shiftx=3, draw=False)
plt.title('query graph and subgraph: Color indicates the label')
plt.show()

plt.figure()
# create some bugs in the nx.draw_networkx, don't know why.
draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
plt.title('test graph: Color indicates the label')
plt.show()

#%%
p1 = ot.unif(len(G1.nodes()))
# ACTUALLY NOT USED IN THE ALGORITHM
p2_nodummy = 1/len(G1.nodes()) * np.ones([N]) * S/N
p2 = np.append(p2_nodummy, 1-sum(p2_nodummy))

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
str_metric = 'adj'
# str_metric = 'shortest_path'

#%% OT-dummy 
thresh=0.004
# WD
fig=plt.figure()
# dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(G1,G2,p1,p2)
dw,log_WD,transp_WD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=0,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('WD coupling')
draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig=plt.figure()
dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# FGWD
alpha=0.5
fig=plt.figure()
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('test graph and FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()



        