# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:05:38 2023

@author: Pandadada
"""
import numpy as np
import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# sys.path.append(os.path.realpath('../lib'))
# sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from lib1.graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from lib1.ot_distances import Fused_Gromov_Wasserstein_distance
import copy
from lib1.data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot

#%% target graph (from dataset)
# dataset_n='mutag' 
# dataset_n='protein' 
# dataset_n='protein_notfull' 
# dataset_n='aids'
# dataset_n='ptc'   # color is not a tuple
# dataset_n='cox2'
dataset_n='bzr'

# path='E:/Master Thesis/OT_sim/FGW-master-4.2.2/FGW-master/data/' #Must link to a folder where all datasets are saved in separate folders
path='E:/Master Thesis/dataset/data/'

# X is consisted of graph objects, label is the label for each graph
X,label=load_local_data(path,dataset_n,wl=0) # using the "wl" option that computes the Weisfeler-Lehman features for each nodes as shown is the notebook wl_labeling.ipynb
# we do not use WL labeling 
NumG = len(X)

Is_info=0

vmin=-5
vmax=20 # the range of color

alpha = 0.5
# fea_metric = 'dirac'
# fea_metric = 'hamming'
fea_metric = 'sqeuclidean'
# str_metric = 'shortest_path'
str_metric = 'adj'

stopThr=1e-09

#%% plot the test graphs:
for i in range(1):
    x=X[i]
    plt.figure()
    draw_rel(x.nx_graph,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
    # plt.title('Test graph: Color indicates the label')
    plt.axis('off')
    plt.show()

#%% create a query graph
# exact subgraph for the first test graph in the dataset
G2_nodummy=Graph() 
x2 = X[0]
G2_nodummy.add_attributes({1:x2.get_attr(1),
                    2:x2.get_attr(2),
                    3:x2.get_attr(3),
                    4:x2.get_attr(4),
                    5:x2.get_attr(5),
                    6:x2.get_attr(6)})  # without dummy

G2_nodummy.add_edge((1,2))
G2_nodummy.add_edge((2,3))
G2_nodummy.add_edge((3,4))
G2_nodummy.add_edge((4,5))
G2_nodummy.add_edge((5,6))
G2_nodummy.add_edge((6,1))

#%% plot the query graph
g2_nodummy=G2_nodummy.nx_graph

plt.figure(figsize=(8,5))
draw_rel(g2_nodummy,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
plt.axis('off')
# plt.title('Query graph: Color indicates the label')
plt.show()
    
#%%
obj=np.zeros(NumG) # dfgw values
for i in range(NumG):
    print(i)
    if label[i]==1:
        dfgw = np.nan
        
    elif label[i]==-1:  # search for all graphs with label "-1"
        x=X[i]
      
        g1=x.nx_graph       
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # # plt.title('Original graph Color indicates the label')
        # plt.axis('off')
        # plt.show()
        
        
        #%% 
        G1=Graph(g1)
        
        G2=copy.deepcopy(G2_nodummy)
        G2.add_attributes({1e6: np.array([0])  })  # add dummy 
        g2=G2.nx_graph
            
        #%%
        if len(x.nodes())<len(G2.nodes()): # test graph does not have enough nodes
            dfgw = np.nan
            continue
            
        #%% plot
        vmin=-5
        vmax=20 # the range of color
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # draw_rel(g2_dummy,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
        # plt.title('Two graphs. Color indicates the label')
        # plt.show()
        
        #%% 
        p1=ot.unif(len(G1.nodes()))
        p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
        p2=np.append(p2_nodummy,1-sum(p2_nodummy))
        
        # dw,transp_WD=Wasserstein_distance(features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
        # # dw=Wasserstein_distance(features_metric='dirac').graph_d(g1,g2)
        thresh=0.002
        # plt.title('WD coupling')
        # draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.show()
        
        ## dgw=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='dirac',method='shortest_path').graph_d(g1,g2)
        # dgw,log_GWD,transp_GWD=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='sqeuclidean',method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
        # plt.figure(figsize=(8,5))
        # plt.title('GWD coupling')
        # draw_transp(G1,G2_dummy,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.show()
        
  
        dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric= fea_metric, method= str_metric, loss_fun='square_loss').graph_d(G1,G2,p1,p2,p2_nodummy, stopThr=stopThr)
        
        # plt.figure(figsize=(8,5))
        # # plt.title('FGWD coupling')
        # draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.axis('off')
        # plt.show()
    
        # plt.close("all")
        
        
        # %% check the features and structure
        if Is_info:
            index = np.argwhere(transp_FGWD[:, 0:-1] > 1e-3)
            # Get the indices that would sort the second column in ascending order
            sort_indices = np.argsort(index[:, 1])
            index = index[sort_indices]
            
            # feature
            Features_source = list(g1._node.values())
            print("Features of subgraph within the source graph:")
            for source in index[:, 0]:  # source is int
                print(Features_source[source])
    
            print("Features of the query graph:")
            Features_target = list(g2_nodummy._node.values())
            for target in index[:, 1]:
                print(Features_target[target])
    
            # structure
            print("Neighbours of source subgraph:")
            Structure_keys = list(g1._node.keys())
            Structure_source = list(g1._adj.values())
            Structure_source2 = {}  # the subgraph within the large graph, but with irrelevant nodes
            for source in index[:, 0]:
                Structure_source2[Structure_keys[source]
                                  ] = Structure_source[source]
    
            temp_keys = list(Structure_source2.keys())
            for key in temp_keys:
                for k in Structure_source2[key].copy():
                    if k not in temp_keys:
                        # delete the irrelevant nodes
                        Structure_source2[key].pop(k, None)
                print(Structure_source2[key])
    
            print("Neighbours of query graph:")
            Structure_target = list(g2_nodummy._adj.values())
            for target in index[:, 1]:
                print(Structure_target[target])
    
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
        
    #%%
    obj[i]=dfgw
            
#%%
# print("all obj values:",obj)

Thre1=1e-5  # threshold for dfgw
index1 = []
obj1 = []
for idx in range(len(obj)) :
    if obj[idx] < Thre1:
        index1.append(idx)
        obj1.append(obj[idx])
print("index1:",index1)
print("obj values from index1:",obj1)

Thre2=1e-3
index2 = []
obj2 = []
for idx in range(len(obj)) :
    if obj[idx] < Thre2:
        index2.append(idx)
        obj2.append(obj[idx])
print("index2:",index2)
print("obj values from index2:",obj2)
