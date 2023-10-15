# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib1'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot

#%% build comunity graphs
def build_comunity_graph(N=30,mu=0,sigma=0.3,pw=0.8):
    v=mu+sigma*np.random.randn(N);
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
         g.add_one_attribute(i,v[i])
         for j in range(N):
             if j != i:
                 r=np.random.rand()
                 if  r<pw:
                     g.add_edge((i,j))
    return g

N=4
mu1=-1.5
pw1=0.8
vmin=-3
vmax=7
np.random.seed(12)
G11=build_comunity_graph(N=N,mu=mu1,sigma=0.8,pw=pw1)

#%% merge community graphs
def merge_graph(g1,g2):  # inputs are nx_graph
    gprime=nx.Graph(g1)
    N0=len(gprime.nodes())
    g2relabel=nx.relabel_nodes(g2, lambda x: x +N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True)) 
    gprime.add_edge(N0-1,N0)
    
    return gprime

g11=G11.nx_graph # initialization with the first comm graph
n=0
Num=[]
g1=merge_graph(g11,g11)
n+=1
    
G1=Graph(g1)

#%% create a query graph
# G2=Graph()
# G2.add_attributes({0:G1.get_attr(0),
#                    1:G1.get_attr(1),
#                    2:G1.get_attr(2),
#                    3:G1.get_attr(3),
#                    4:G1.get_attr(4)})  # without dummy

# G2.add_edge((0,1))
# G2.add_edge((0,2))
# G2.add_edge((0,4))
# G2.add_edge((1,2))
# G2.add_edge((1,4))
# G2.add_edge((1,3))
# G2.add_edge((2,4))
# G2.add_edge((3,4))

g2_nodummy=g11 # initialization with the first comm graph
    
G2_nodummy=Graph(g2_nodummy)

G2=copy.deepcopy(G2_nodummy)
G2.add_attributes({100: 0 })  # add dummy 

#%%  The followings are fixed
g1 = G1.nx_graph
g2 = G2.nx_graph

plt.figure(figsize=(8,5))
draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
draw_rel(g2,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
plt.title('Two graphs. Color indicates the label')
plt.show()

#%% weights and feature metric
p1=ot.unif(len(G1.nodes()))
p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
p2=np.append(p2_nodummy,1-sum(p2_nodummy))

# fea_metric = 'dirac'
# fea_metric = 'hamming'
fea_metric = 'sqeuclidean'

#%%
thresh=0.004
fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'
loss_fun='square_loss'
# loss_fun = 'kl_loss'

# FGWD
alpha=0.9
fig=plt.figure()
dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
    alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun=loss_fun).graph_d(G1, G2, p1, p2, p2_nodummy)
plt.title('FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()


