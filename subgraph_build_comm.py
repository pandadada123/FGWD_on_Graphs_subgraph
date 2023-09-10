# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_0.0'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
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

N=5
mu1=-1.5
mu2=1.5
mu3=3
pw1=0.8
pw2=0.3
pw3=0.8
vmin=-3
vmax=7
np.random.seed(12)
G11=build_comunity_graph(N=N,mu=mu1,sigma=0.8,pw=pw1)
G12=build_comunity_graph(N=N,mu=mu2,sigma=0.8,pw=pw2)
G13=build_comunity_graph(N=N,mu=mu3,sigma=0.8,pw=pw3)
com_graph={1:G11,2:G12,3:G13}

#%% merge community graphs
def merge_graph(g1,g2):  # inputs are nx_graph
    gprime=nx.Graph(g1)
    N0=len(gprime.nodes())
    g2relabel=nx.relabel_nodes(g2, lambda x: x +N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True)) 
    gprime.add_edge(N0-1,N0)
    
    return gprime

g1=G11.nx_graph # initialization with the first comm graph
n=0
Num=[]
while n<=3:
    num=np.random.randint(1,4) # randomly generate a number within [1,2,3]
    Num=np.append(Num,num)
    g1=merge_graph(g1, com_graph[num].nx_graph)
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

g2_nodummy=G11.nx_graph # initialization with the first comm graph
n=0
while n<=1:
    g2_nodummy=merge_graph(g2_nodummy, G11.nx_graph) # merge 2 G11 graphs
    n+=1
    
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
# WD
fig=plt.figure(figsize=(10,8))
dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(G1,G2,p1,p2)
# dw,log_WD,transp_WD=Fused_Gromov_Wasserstein_distance(alpha=0,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('WD coupling')
draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig=plt.figure(figsize=(10,8))
dgw,log_GWD,transp_GWD=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# FGWD
alpha=0.99
fig=plt.figure(figsize=(10,8))
dfgw,log_FGWD,transp_FGWD=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
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

