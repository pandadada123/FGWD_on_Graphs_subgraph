# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:49:52 2023

@author: Pandadada
"""

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
# sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from lib1.graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from lib1.ot_distances import Fused_Gromov_Wasserstein_distance

import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot

'''G is Graph object
   g is nx_graph
'''

stopThr = 1e-09

#%%
# first graph of three nodes
G11=Graph() 
G11.add_attributes({0:1,1:7,2:5})    # add color to nodes
G11.add_edge((0,1))
G11.add_edge((1,2))
G11.add_edge((0,2))

G11.add_edge((0,0))
G11.add_edge((1,1))
G11.add_edge((2,2))

# second graph of two nodes
G12=Graph() 
G12.add_attributes({0:2,1:4})

#%%
vmin=0
vmax=9  # the range of color

#%%
G1 = copy.deepcopy(G11)
G2_nodummy = copy.deepcopy(G12)
G2 = copy.deepcopy(G12)
G2.add_attributes({5:1})

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
str_metric = 'adj'
# str_metric = 'shortest_path'

#%%
thresh=0.004
# WD
fig=plt.figure(figsize=(10,8))
# dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(G1,G2,p1,p2)
dw,log_WD,transp_WD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=0,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy, stopThr=stopThr)
plt.title('WD coupling')
draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig=plt.figure(figsize=(10,8))
dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy, stopThr=stopThr)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# FGWD
alpha=0.5
fig=plt.figure(figsize=(10,8))
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method=str_metric,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy, stopThr=stopThr)
plt.title('FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

print("WD:", dw)
print("GWD:", dgw)
print("FGWD:", dfgw)
