# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:03:35 2023

@author: Pandadada
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import KEGGutils as kg
import networkx as nx

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.1'))

import numpy as np
from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from ot_distances import Fused_Gromov_Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot


kg.delete_cached_files()

# pathway1 = kg.KEGGpathway(pathway_id = "hsa05215")  # API for KEGG, "pathway type + index"
P1 = kg.KEGGpathway(pathway_id = "hsa05224")  # API for KEGG, "pathway type + index"

# print(pathway.title)
# print(pathway.weblink)

# pathway.tree
# pathway.tree.getroot().attrib

# pathway.genes['38'] # for dictionary "gene"
# # pathway.relations['46to38'] # for dictionary "relations"

# node_ids = pathway.relations['46to38']['node_ids']

# pathway1.draw()

P2_nodummy = kg.KEGGpathway(pathway_id = "hsa04012")
# pathway2.draw()


#%% 
def KEGGpathwayToGraph(P):
    Nodes=P.nodes()
    G=Graph()
    for node in Nodes:
        feature = P._node[node]['label']
        G.add_attributes({node : feature})
        edges = P._adj[node].keys()
        for edge in edges:
            G.add_edge((node,edge))
        
    return G
        
    
#%%
# P2=copy.deepcopy(P2_nodummy)
# P2.add_node(100)  # add dummy 
#%%
# G1 = copy.deepcopy(P1)
# G2_nodummy = copy.deepcopy(P2_nodummy)
# G2 = copy.deepcopy(P2)

G1=KEGGpathwayToGraph(P1)
G2_nodummy=KEGGpathwayToGraph(P2_nodummy)
G2=copy.deepcopy(G2_nodummy)
G2.add_attributes({len(G2.nodes()): '0' })  # add dummy 

g1=G1.nx_graph
g2=G2.nx_graph
#%% weights and feature metric
p1=ot.unif(len(G1.nodes()))
p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
p2=np.append(p2_nodummy,1-sum(p2_nodummy))

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
vmin=0
vmax=9  # the range of color
thresh=0.004
# FGWD
alpha=0
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
# fig=plt.figure()
# plt.title('FGWD coupling')
# draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()