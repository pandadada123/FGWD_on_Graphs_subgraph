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

# API for KEGG, "pathway type + index"
# P1 = kg.KEGGpathway(pathway_id = "hsa05224")  # cancer
# P1 = kg.KEGGpathway(pathway_id = "hsa05010")  # Alzheimer 
P1 = kg.KEGGpathway(pathway_id = "hsa05012")  # Parkinson
# PP1 = kg.KEGGgraph(pathway_id = "D11976") # not work

# print(pathway.title)
# print(pathway.weblink)

# pathway.tree
# pathway.tree.getroot().attrib

# pathway.genes['38'] # for dictionary "gene"
# # pathway.relations['46to38'] # for dictionary "relations"

# node_ids = pathway.relations['46to38']['node_ids']

# pathway1.draw()

# P2_nodummy = kg.KEGGpathway(pathway_id = "hsa04012")
# pathway2.draw()


#%% 
def KEGGpathwayToGraph(P):
    Nodes=P.nodes()
    G=Graph()
    for node in Nodes:
        if '_' in node:
            continue
        else:
            feature = P._node[node]['label']
            G.add_attributes({node : feature})
            # G.add_attributes({i : feature})
            edges = P._adj[node].keys()
            for edge in edges:
                if '_' in edge:
                    continue
                else:
                    G.add_edge((node,edge))
                    # G.add_edge((i,edge))
    # change the keys to numbers
    
    return G
        
    
#%%
# P2=copy.deepcopy(P2_nodummy)
# P2.add_node(100)  # add dummy 
#%%
# G1 = copy.deepcopy(P1)
# G2_nodummy = copy.deepcopy(P2_nodummy)
# G2 = copy.deepcopy(P2)

#%% build a subgraph (ERK)
# G2_nodummy = Graph()
# G2_nodummy.add_attributes({'0':'HRAS',
#                   '1':'ARAF',
#                   '2':'MEK1',
#                   '3':'ERK'})
# G2_nodummy.add_edge(('0','1'))
# G2_nodummy.add_edge(('1','2'))
# G2_nodummy.add_edge(('2','3'))

#%% build a subgraph (UPR): only need feature to find 
G2_nodummy = Graph()
G2_nodummy.add_attributes({'0':'SNCA', '1':'BIP', '2':'ATF6', '3':'IRE1a',
                            '4':'PERK', '5':'CHOP','6':'XBP1', '7':'EIF2A',
                            '8':'ATF4', '9':'CHOP'})
G2_nodummy.add_edge(('0','1'))
G2_nodummy.add_edge(('1','2'))
G2_nodummy.add_edge(('1','3'))
G2_nodummy.add_edge(('1','4'))
G2_nodummy.add_edge(('2','5'))
G2_nodummy.add_edge(('3','6'))
G2_nodummy.add_edge(('4','7'))
G2_nodummy.add_edge(('7','8'))
G2_nodummy.add_edge(('8','9'))

#%%
G1=KEGGpathwayToGraph(P1)
# G2_nodummy=KEGGpathwayToGraph(P2_nodummy)
G2=copy.deepcopy(G2_nodummy)
G2.add_attributes({len(G2.nodes()): '0' })  # add dummy 

g1=G1.nx_graph
g2=G2.nx_graph
g2_nodummy=G2_nodummy.nx_graph

#%% weights and feature metric
p1=ot.unif(len(G1.nodes()))
p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
p2=np.append(p2_nodummy,1-sum(p2_nodummy))

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
# str_metrc = 'shortest_path'
str_metric = 'adj'
vmin=0
vmax=9  # the range of color
thresh=0.004
# FGWD
alpha=0.2
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric=fea_metric, method= str_metric ,loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
# fig=plt.figure()
# plt.title('FGWD coupling')
# draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()

#%% get the subgraoh from transp_FGWD
index = np.argwhere(transp_FGWD[:,0:-1]> 1e-5)
sort_indices = np.argsort(index[:, 1]) # Get the indices that would sort the second column in ascending order
index = index[sort_indices]

# feature
Features_source = list(g1._node.values())
print ("Features of subgraph within the source graph:")
for source in index[:,0]:  # source is int
    print (Features_source[source])
    
print ("Features of the query graph:")
Features_target = list(g2_nodummy._node.values())
for target in index[:,1]:
    print (Features_target[target])
    
# structure 
print ("Neighbours of source subgraph:")
Structure_keys = list(g1._node.keys())
Structure_source = list(g1._adj.values())
Structure_source2 = {}  # the subgraph within the large graph, but with irrelevant nodes
for source in index[:,0]:
    Structure_source2[Structure_keys[source]]=Structure_source[source]
    
temp_keys = list(Structure_source2.keys())
for key in temp_keys:
    for k in Structure_source2[key].copy():
        if k not in temp_keys:
            Structure_source2[key].pop(k, None) # delete the irrelevant nodes
    print (Structure_source2[key])

print ("Neighbours of query graph:")
Structure_target = list(g2_nodummy._adj.values())
for target in index[:,1]:
    print (Structure_target[target])
    

# Adj matrix 
def generate_adjacency_matrix(graph_dict):
    # Get all unique nodes from the dictionary keys
    nodes = list(graph_dict.keys())
    num_nodes = len(nodes)
    
    # Initialize an empty adjacency matrix with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Iterate over the graph dictionary
    for node, connections in graph_dict.items():
        # Get the index of the current node
        node_index = nodes.index(node)
        
        # Iterate over the connected nodes
        for connected_node in connections.keys():
            # Get the index of the connected node
            connected_node_index = nodes.index(connected_node)
            
            # Set the corresponding entry in the adjacency matrix to 1
            adjacency_matrix[node_index][connected_node_index] = 1
    
    return adjacency_matrix


adjacency_subgraph = generate_adjacency_matrix(Structure_source2)
print("Adjacency matrix within the source graph")
print(adjacency_subgraph)

adjacency_query = generate_adjacency_matrix(g2_nodummy._adj)
print("Adjacency matrix of query graph")
print(adjacency_query)
    