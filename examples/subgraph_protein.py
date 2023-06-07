# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:05:38 2023

@author: Pandadada
"""
import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance
import copy
from data_loader import load_local_data,histog,build_noisy_circular_graph
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
# X is consisted of graph objects
X,label=load_local_data(path,dataset_n,wl=0) # using the "wl" option that computes the Weisfeler-Lehman features for each nodes as shown is the notebook wl_labeling.ipynb
# we do not use WL labeling 
# x=X[2]

Is_info=1

vmin=-5
vmax=20 # the range of color

# for i in range(663,673):
#     x=X[i]
#     plt.figure(figsize=(8,5))
#     draw_rel(x.nx_graph,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
#     plt.title('Original graph Color indicates the label')
#     plt.show()

#%% create a query graph
G2_nodummy=Graph()
# %% protein 
# G2_nodummy.add_attributes({1:x.get_attr(1),
#                   12:x.get_attr(12),
#                   23:x.get_attr(23),
#                   33:x.get_attr(33),
#                   31:x.get_attr(31),
#                   41:x.get_attr(41)})  # without dummy

# G2_nodummy.add_edge((1,12))
# G2_nodummy.add_edge((1,33))
# G2_nodummy.add_edge((1,23))
# G2_nodummy.add_edge((12,23))
# G2_nodummy.add_edge((12,33))
# G2_nodummy.add_edge((23,31))
# G2_nodummy.add_edge((23,41))
# G2_nodummy.add_edge((33,41))
# G2_nodummy.add_edge((33,31))

# G2_nodummy.add_attributes({1:[23.0],
#                     2:[22.0],
#                     3:[3.0],
#                     4:[3.0],
#                     5:[3.0],
#                     6:[3.0]})  # without dummy

# G2_nodummy.add_edge((1,2))
# G2_nodummy.add_edge((1,3))
# G2_nodummy.add_edge((1,4))
# G2_nodummy.add_edge((2,3))
# G2_nodummy.add_edge((2,4))
# G2_nodummy.add_edge((3,5))
# G2_nodummy.add_edge((3,6))
# G2_nodummy.add_edge((4,5))
# G2_nodummy.add_edge((4,6))

# G2_nodummy.add_attributes({1:[8.0],
#                     2:[3.0],
#                     3:[5.0],
#                     4:[4.0],
#                     5:[7.0],
#                     6:[7.0]})  # without dummy
# G2_nodummy.add_edge((1,2))
# G2_nodummy.add_edge((1,3))
# G2_nodummy.add_edge((2,3))
# G2_nodummy.add_edge((2,4))
# G2_nodummy.add_edge((3,4))
# G2_nodummy.add_edge((3,5))
# G2_nodummy.add_edge((4,5))
# G2_nodummy.add_edge((4,6))
# G2_nodummy.add_edge((5,6))

# x2 = X[16]
# G2_nodummy.add_attributes({1:x2.get_attr(961),
#                     2:x2.get_attr(944),
#                     3:x2.get_attr(945),
#                     4:x2.get_attr(946),
#                     5:x2.get_attr(947),
#                     6:x2.get_attr(962)})  # without dummy
# G2_nodummy.add_edge((1,2))
# G2_nodummy.add_edge((1,3))
# G2_nodummy.add_edge((2,3))
# G2_nodummy.add_edge((2,4))
# G2_nodummy.add_edge((3,4))
# G2_nodummy.add_edge((3,5))
# G2_nodummy.add_edge((4,5))
# G2_nodummy.add_edge((4,6))
# G2_nodummy.add_edge((5,6))

#%% bzr 1 
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

# G2_nodummy.add_edge((1,3))
# G2_nodummy.add_edge((1,5))
# G2_nodummy.add_edge((2,4))
# G2_nodummy.add_edge((2,5))
# G2_nodummy.add_edge((2,6))
# G2_nodummy.add_edge((3,5))
# G2_nodummy.add_edge((3,6))
# G2_nodummy.add_edge((4,6))

# G2_nodummy.add_attributes({1:x.get_attr(1),
#                     2:x.get_attr(2),
#                     3:x.get_attr(3),
#                     4:x.get_attr(4),
#                     5:x.get_attr(5),
#                     6:x.get_attr(6),
#                     7:x.get_attr(7),
#                     8:x.get_attr(8),
#                     9:x.get_attr(9),
#                     10:x.get_attr(10),
#                     11:x.get_attr(11),})  # without dummy

# G2_nodummy.add_edge((1,2))
# G2_nodummy.add_edge((2,3))
# G2_nodummy.add_edge((3,4))
# G2_nodummy.add_edge((4,5))
# G2_nodummy.add_edge((5,6))
# G2_nodummy.add_edge((6,1))

# G2_nodummy.add_edge((4,7))
# G2_nodummy.add_edge((7,8))
# G2_nodummy.add_edge((8,9))
# G2_nodummy.add_edge((9,10))
# G2_nodummy.add_edge((10,11))
# G2_nodummy.add_edge((11,5))
#%%
g2_nodummy=G2_nodummy.nx_graph

plt.figure(figsize=(8,5))
draw_rel(g2_nodummy,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
plt.axis('off')
plt.show()
    
    #%%
obj=[]
# for i in range(663,1113):
# for i in range(len(X)):
for i in range(1):
    # print(i)
    # if label[i]==1:
    #     dfgw = np.nan
        
    # elif label[i]==-1:
    #     x=X[i]
       
        x=X[2]
        # x = X[5]
        # plt.figure(figsize=(8,5))
        # draw_rel(X[i].nx_graph,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # plt.title('Original graph Color indicates the label')
        # plt.show()
        
        # x.add_edge((1,3))
        # x.add_edge((1,5))
        # x.add_edge((2,4))
        # x.add_edge((2,5))
        # x.add_edge((2,6))
        # x.add_edge((3,5))
        # x.add_edge((3,6))
        # x.add_edge((4,6))
        
        g1=x.nx_graph
        # x2=X[1].nx_graph
        # x188=X[187].nx_graph
        # x1Graph=Graph(x1)
        # G1=Graph(g1)
        
    
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # # plt.title('Original graph Color indicates the label')
        # plt.axis('off')
        # plt.show()
        
    
        #%% [DO NOT NEED]
        # def change(graph,tohash=True):
        def change(graph):  # change attribute of list to array AND change the node indexes to start from 0 [no use any nore]
        
            final_graph=nx.Graph(graph)
            
            dict_values={} # pas sûr d'ici niveau de l'ordre des trucs;  not sure about here level of stuff order
            for k,v in final_graph.nodes().items():  # k: index of node, v: attribute of node
            
                # hashed=sorted([str(x) for x in v.values()], key=len)
            
                # if tohash :
                #     dict_values[k]=np.array([hash(x) for x in hashed])
                # else:
                #     dict_values[k]=np.array(hashed)
            
                attr = v.get('attr_name')
                dict_values[k] = np.array(attr)
                
            graph2=nx.Graph(graph)
            # GG2 =  Graph()
            # for k,v in dict_values:
            #     GG2.add_aatributes i
            
            nx.set_node_attributes(graph2,dict_values,'attr_name')   # dict_values are 3-dim values; graph2 is 3-dim features 
        
            graph2._node=dict((key-1, value) for (key, value) in graph2._node.items())    
            graph2._adj=dict((key-1, value) for (key, value) in graph2._adj.items())  
            
            # for i in range(len(graph2._node)):
            for i,v in graph2.nodes().items():
                graph2._adj[i]=dict((key-1, value) for (key, value) in graph2._adj[i].items())  
            
            return graph2   
            
        # G1 = Graph(change(g1))
        # g1=G1.nx_graph
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # plt.title('Original graph Color indicates the label')
        # plt.show()
        
        #%%
        G1=Graph(g1)
        G2=copy.deepcopy(G2_nodummy)
        # G2_dummy.add_attributes({100: np.array ( [100]*(len(x.get_attr(1)))  )  })  # add dummy 
        G2.add_attributes({1e6: np.array([0])  })  # add dummy 
        # G2_dummy.add_attributes({len(G2.nodes()): [0] })  # add dummy 
        g2=G2.nx_graph
        
        # G2_dummy = Graph(change(g2_dummy))
        # G2 = Graph(change(g2))
        # g2=G2.nx_graph
        #%%
        # g1=G1.nx_graph
        # g2_nodummy=G2_nodummy.nx_graph
        # g2=G2.nx_graph
            
        
        #%% plot
        vmin=-5
        vmax=20 # the range of color
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # draw_rel(g2_dummy,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
        # plt.title('Two graphs. Color indicates the label')
        # plt.show()
        
        #%% FGWD
        # G1=Graph(g1)
        # G2=Graph(g2)
        # G2_dummy=Graph(g2_dummy)
        
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
        
        
        #%%
        # We want alpha that maximize FGWD
        # plt.figure()
        # alld=[]
        # Alpha=np.linspace(0,1,20)
        # for alpha in Alpha:
        #     dfgw,log_FGWD,transp_FGWD=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric='sqeuclidean',method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2_dummy,p1,p2,p2_nodummy)
        #     alld.append(dfgw)
        # plt.plot(Alpha,alld)
        # plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(Alpha[np.argmax(alld)]))
        # plt.xlabel('Alpha')
        # plt.xlabel('FGW dist')
        # plt.show()
        
        # alpha=Alpha[  alld.index(max(alld))  ]  # get the optimal alpha
        
        alpha = 0.2
        # fea_metric = 'dirac'
        # fea_metric = 'hamming'
        fea_metric = 'sqeuclidean'
        # str_metric = 'shortest_path'
        str_metric = 'adj'
        dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric= fea_metric, method= str_metric, loss_fun='square_loss').graph_d(G1,G2,p1,p2,p2_nodummy)
        
        plt.figure(figsize=(8,5))
        # plt.title('FGWD coupling')
        draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        plt.axis('off')
        plt.show()
    
        # plt.close("all")
        
        #%%
        if len(x.nodes())<len(G2.nodes()):
            dfgw = np.nan
            
        obj.append(dfgw)
        
        print (i)
    
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
print("all obj values:",obj)

Thre1=1e-5
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
