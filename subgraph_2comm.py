# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:49:52 2023

@author: Pandadada
"""

# Finding subgraph

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from ot_distances import Fused_Gromov_Wasserstein_distance

import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot

'''G is Graph object
   g is nx_graph
'''
#%% 
G11=Graph() # community 1
G11.add_attributes({0:1,1:7,2:5,3:3,4:9})    # add color to nodes
G11.add_edge((0,1))
G11.add_edge((1,2))
G11.add_edge((2,3))
G11.add_edge((3,4))
G11.add_edge((4,0))
# g1.add_edge((0,2))
# g1.add_edge((1,3))
G11.add_edge((0,2))
G11.add_edge((0,3))
G11.add_edge((2,4))
G11.add_edge((1,3))
G11.add_edge((1,4))

G12=Graph() # community 2
G12.add_attributes({0:1,1:7,2:5,3:3,4:9})
G12.add_edge((0,3))
G12.add_edge((1,3))
G12.add_edge((2,3))
G12.add_edge((4,3))

# N=5 # five different colors/nodes in each subgra
# mu1=-1.5
# mu2=1.5
vmin=0
vmax=9  # the range of color
# np.random.seed(12)
# g1=build_comunity_graph(N=N,mu=mu1,sigma=0.8,pw=0.5)
# g2=build_comunity_graph(N=N,mu=mu2,sigma=0.8,pw=0.5)


def merge_graph(g1,g2):  # inputs are nx_graph
    gprime=nx.Graph(g1)
    N0=len(gprime.nodes())
    g2relabel=nx.relabel_nodes(g2, lambda x: x +N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True)) 
    gprime.add_edge(N0-1,N0)
    
    return gprime

G1 = Graph ( merge_graph(G11.nx_graph,G12.nx_graph) )  # Graph including 2 communities
G2_nodummy = copy.deepcopy(G11) # version without dummy node
G2 = copy.deepcopy(G11)
G2.add_attributes({5:1}) # add a dummy node to G2

#%% example from SAGA
# G11=Graph() # community 1
# G11.add_attributes({0:0,1:1,2:2,3:3,4:4,5:5})    # add color to nodes
# G11.add_edge((0,1))
# G11.add_edge((1,2))
# G11.add_edge((2,3))
# G11.add_edge((2,4))
# G11.add_edge((2,5))

# G12=Graph() # community 2
# G12.add_attributes({0:2,1:3,2:4,3:5,4:6,5:7})
# G12.add_edge((0,3))
# G12.add_edge((0,4))
# G12.add_edge((4,5))
# G12.add_edge((4,3))
# G12.add_edge((2,5))
# G12.add_edge((1,5))

# G1 = copy.deepcopy(G11)
# G2_nodummy = copy.deepcopy(G12)
# G2 = copy.deepcopy(G2_nodummy)
# G2.add_attributes({100:0}) # add a dummy node to G2

# # N=5 # five different colors/nodes in each subgra
# # mu1=-1.5
# # mu2=1.5
# vmin=0
# vmax=9  # the range of color

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

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'

#%%
thresh=0.004
# WD
fig=plt.figure(figsize=(10,8))
# dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(G1,G2,p1,p2)
dw,log_WD,transp_WD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=0,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('WD coupling')
draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig=plt.figure(figsize=(10,8))
dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# FGWD
alpha=0.2
fig=plt.figure(figsize=(10,8))
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

#%% FGWD, find alpha
alld=[]
x=np.linspace(0,1,10)
for alpha in x:
    d,log,transp=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
    alld.append(d)
plt.plot(x,alld)
plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(x[np.argmax(alld)]))
plt.xlabel('Alpha')
plt.xlabel('FGW dist')
plt.show()

# optimal matching
fig=plt.figure(figsize=(10,8))
thresh=0.004
alpha_opt=x [ alld.index(max(alld)) ]
dfgw_opt,log_FGWD_opt,transp_FGWD_opt=Fused_Gromov_Wasserstein_distance(alpha=alpha_opt,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
# d=dfgw.graph_d(g1,g2)
# plt.title('FGW coupling, dist : '+str(np.round(dfgw,3)),fontsize=15)
plt.title('FGW coupling, alpha = opt')
draw_transp(G1,G2,transp_FGWD_opt,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# print('Wasserstein distance={}, Gromov distance={} \nFused Gromov-Wasserstein distance for alpha {} = {}'.format(dw,dgw,alpha,dfgw))

