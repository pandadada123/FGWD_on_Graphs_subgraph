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
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib2'))

import numpy as np
from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from ot_distances import Fused_Gromov_Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot
import time

kg.delete_cached_files()

# API for KEGG, "pathway type + index"
P1 = kg.KEGGpathway(pathway_id = "hsa05224")  # cancer
# P1 = kg.KEGGpathway(pathway_id = "hsa05010")  # Alzheimer 
# P1 = kg.KEGGpathway(pathway_id = "hsa05012")  # Parkinson
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
G2_nodummy = Graph()
G2_nodummy.add_attributes({'0':'HRAS',
                  '1':'ARAF',
                  '2':'MEK1',
                  '3':'ERK'})
G2_nodummy.add_edge(('0','1'))
G2_nodummy.add_edge(('1','2'))
G2_nodummy.add_edge(('2','3'))

#%% build a subgraph (UPR): only need feature to find 
# G2_nodummy = Graph()
# G2_nodummy.add_attributes({'0':'SNCA', '1':'BIP', '2':'ATF6', '3':'IRE1a',
#                             '4':'PERK', '5':'CHOP','6':'XBP1', '7':'EIF2A',
#                             '8':'ATF4', '9':'CHOP'})
# G2_nodummy.add_edge(('0','1'))
# G2_nodummy.add_edge(('1','2'))
# G2_nodummy.add_edge(('1','3'))
# G2_nodummy.add_edge(('1','4'))
# G2_nodummy.add_edge(('2','5'))
# G2_nodummy.add_edge(('3','6'))
# G2_nodummy.add_edge(('4','7'))
# G2_nodummy.add_edge(('7','8'))
# G2_nodummy.add_edge(('8','9'))

#%%
G1=KEGGpathwayToGraph(P1)
# G2_nodummy=KEGGpathwayToGraph(P2_nodummy)
G2=copy.deepcopy(G2_nodummy)
G2.add_attributes({len(G2.nodes()): '0' })  # add dummy 

g1=G1.nx_graph
g2=G2.nx_graph
g2_nodummy=G2_nodummy.nx_graph

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metrc = 'shortest_path'
str_metric = 'adj'
loss_fun='square_loss'

alpha1=0
alpha2=0.5

thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-2
thre3 = 0.05
epsilon = thre1

K=3

#%%
def find_center_with_smallest_longest_hops(graph):
    min_longest_hops = float('inf') 
    center_node_query = None
    
    for node in graph.nodes():
        longest_hops = max(nx.shortest_path_length(graph, source=node).values())
        
        if longest_hops < min_longest_hops:
            min_longest_hops = longest_hops
            center_node_query = node
            
    longest_path_center = min_longest_hops
    
    return longest_path_center

start_time_center = time.time()
g2_longest_path_from_center = find_center_with_smallest_longest_hops(g2_nodummy)
end_time_center = time.time()
time_center = end_time_center - start_time_center  # almost zero

#%% Sliding window: go over every node in target
g1_sliding_list=[]        
G1_sliding_list = []
dw_sliding_list = []
dfgw_sliding_list  = []
transp_FGWD_sliding_list = []

ii=0
sliding_time = 0
for center_node in g1.nodes():
    print(ii)
    ii+=1
    
    # Using h-diameter neighborhood hops to create sliding subgraph
    def create_h_hop_subgraph(graph, center_node, h):
        subgraph_nodes = set([center_node])
        neighbors = set([center_node])
    
        for _ in range(h):
            new_neighbors = set()
            for node in neighbors:
                new_neighbors.update(graph.neighbors(node))
            subgraph_nodes.update(new_neighbors)
            neighbors = new_neighbors
            
        h_hop_subgraph = graph.subgraph(subgraph_nodes).copy()
    
        return h_hop_subgraph
    
    # induced_subgraph = create_h_hop_subgraph(g1, center_node, h=math.ceil(g2_diameter/2))  # sometimes could not include the subgraph in the big graph
    # induced_subgraph = create_h_hop_subgraph(g1, center_node, h=math.ceil(g2_diameter))
    start_time=time.time()
    time0=time.time()
    g1_sliding = create_h_hop_subgraph(g1, center_node, h = g2_longest_path_from_center)
    G1_sliding = Graph(g1_sliding)
    time1=time.time()
    if len(G1_sliding.nodes()) < len(G2_nodummy.nodes()):  
        print("The sliding subgraph did not get enough nodes.")
        continue # go to the next sliding subgraph
    
    G2 = copy.deepcopy(G2_nodummy)
    
    G2.add_attributes({len(G2.nodes()): "0"})  # add dummy            
    time2=time.time()

    # %% plot the graphs
    # vmin = 0
    # vmax = 9  # the range of color

    # plt.figure(figsize=(8, 5))
    # # create some bugs in the nx.draw_networkx, don't know why.
    # draw_rel(g1_sliding, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
    # draw_rel(g2_nodummy, vmin=vmin, vmax=vmax,
    #           with_labels=True, shiftx=3, draw=False)
    # plt.title('Sliding subgraph and query graph: Color indicates the label')
    # plt.show()

    # %% weights and feature metric
    p1 = ot.unif(len(G1_sliding.nodes()))
    # ACTUALLY NOT USED IN THE ALGORITHM
    p2_nodummy = 1/len(G1_sliding.nodes()) * np.ones([len(G2_nodummy.nodes())])
    p2 = np.append(p2_nodummy, 1-sum(p2_nodummy))
    
    # p1 = np.ones(len(G1_sliding.nodes()))
    # # ACTUALLY NOT USED IN THE ALGORITHM
    # p2_nodummy = np.ones([len(G2_nodummy.nodes())])
    # p2 = np.append(p2_nodummy, sum(p1)-sum(p2_nodummy))
    
    time3=time.time()
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
    
    #%% Wasserstein filtering
    # epsilon = thre1
    # alpha = 0
    dw, log_WD, transp_WD, M, C1, C2  = Fused_Gromov_Wasserstein_distance(
        alpha=alpha1, features_metric=fea_metric, method=str_metric, loss_fun=loss_fun).graph_d(G1_sliding, G2, p1, p2, p2_nodummy)
    time4=time.time()
    if dw > epsilon:
        print("filter out")
        continue # go to the next sliding subgraph
    time5=time.time()
    # %% FGWD
    # alpha = 0.5
    dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
        alpha=alpha2, features_metric=fea_metric, method=str_metric, loss_fun=loss_fun).graph_d(G1_sliding, G2, p1, p2, p2_nodummy)
    time6=time.time()
    end_time = time.time()
    # %% keep an record of the successful sliding subgraphs and their dw
    dw_sliding_list.append(dw)
    
    g1_sliding_list.append(g1_sliding)                   
    G1_sliding_list.append(G1_sliding)
        
    # keep an record of the successful dfgw and transp
                   
    dfgw_sliding_list.append(dfgw)
    
    transp_FGWD_sliding_list.append(transp_FGWD)
    
    sliding_time += end_time - start_time 
        
#%% 

# Get the indices sorted by their corresponding values
sorted_indices = sorted(range(len(dfgw_sliding_list)), key=lambda i: dfgw_sliding_list[i])
    
# Take the top-k of these indices
topk_indices = sorted_indices[:K]

# Extract the top-k minimum values using these indices
topk_mins = [dfgw_sliding_list[i] for i in topk_indices]

# dgfw_sliding_min = min(dfgw_sliding_list)

# min_index = dfgw_sliding_list.index(dgfw_sliding_min)

transp_FGWD_sliding_min = [transp_FGWD_sliding_list[i] for i in topk_indices]
g1_sliding_min = [g1_sliding_list[i] for i in topk_indices]
G1_sliding_min = [G1_sliding_list[i] for i in topk_indices]
dw_sliding_min = [dw_sliding_list[i] for i in topk_indices]

# vmin = 0
# vmax = 9  # the range of color
# fig = plt.figure(figsize=(10, 8))
# plt.title('Optimal FGWD coupling')
# draw_transp(G1_sliding_min, G2, transp_FGWD_sliding_min, shiftx=2, shifty=0.5, thresh=thresh,  # check the node order when drawing
#             swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
# plt.show()

#%% get the subgraoh from transp_FGWD
for k in range(K):
    print(k)
    
    index = np.argwhere(transp_FGWD_sliding_min[k][:,0:-1]> 1e-5)
    sort_indices = np.argsort(index[:, 1]) # Get the indices that would sort the second column in ascending order
    index = index[sort_indices]
    
    g1 = g1_sliding_min[k]
    
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
    