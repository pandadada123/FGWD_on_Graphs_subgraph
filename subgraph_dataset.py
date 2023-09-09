# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""


import numpy as np
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# sys.path.append(os.path.realpath('../lib'))
# sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib1'))
# sys.path.append(os.path.realpath('C:/Users/Thinkpad/Desktop/temp/lib1'))


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
import math

import pickle


thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-4       
        
Is_fig = 0
Is_info = 0
Is_create_query = 0
Is_fea_noise = 1
Is_str_noise = 1

mean_fea = 0
std_fea = 0.1
str_mean = 0
str_std = 0.1

Num = 1 # number of random graphs
# fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'

DFGW_set = []
Percent1 = []
Percent2 = []
Mean = []
STD = []
Lower = []
Upper = []

# g1 = pickle.load(open('\home\pan\dataset\data_pickle\imdb\G_imdb.pickle', 'rb'))
# g2_nodummy = pickle.load(open('\home\pan\dataset\data_pickle\imdb\Q_imdb.pickle', 'rb'))
g1 = pickle.load(open('E:\Master Thesis\dataset\data_pickle\yago\G_yago.pickle', 'rb'))
g2_nodummy = pickle.load(open('E:\Master Thesis\dataset\data_pickle\yago\Q_yago.pickle', 'rb'))

plt.close("all")
        
NumQ = 1 # Number of query graphs

#%% add noise to the query
def add_noise_to_query(g,mean_fea,std_fea,str_mean,str_std,
                       Is_fea_noise,Is_str_noise):    
    if Is_fea_noise: # Add label noise
        if fea_metric == 'jaccard':
            for node in g.nodes():
                current_string = g.nodes[node]['attr_name']
                # Convert the input string to a list of Unicode code points
                code_points = [ord(char) for char in current_string]
            
                # Apply Gaussian noise to each code point
                noisy_code_points = [
                    int(round(code + np.random.normal(mean_fea, std_fea)))
                    for code in code_points
                ]
            
                # Ensure that code points are within valid Unicode range (32 to 126)
                noisy_code_points = [
                    min(max(code, 32), 126)
                    for code in noisy_code_points
                ]
            
                # Convert the noisy code points back to a string
                noisy_string = ''.join([chr(code) for code in noisy_code_points])
                
                g.nodes[node]['attr_name'] = noisy_string

        elif fea_metric == 'dirac':
            for node in g.nodes():
                current_value = g.nodes[node]['attr_name']
                noise = np.random.normal(mean_fea, std_fea)
                new_value = current_value + noise
                g.nodes[node]['attr_name'] = round(new_value)  # still int value
            
    if Is_str_noise: # Add structural noise
        # Generate random values for edge insertions and deletions
        num_insertions = max(0, int(np.random.normal(str_mean/2, str_std)))
        num_deletions = max(0, int(np.random.normal(str_mean/2, str_std)))
        
        # Structural noise: Edge insertions
        for _ in range(num_insertions):
            node1, node2 = random.sample(g.nodes(), 2)
            if not g.has_edge(node1, node2):
                g.add_edge(node1, node2)
        
        # Structural noise: Edge deletions
        for _ in range(num_deletions):
            edges = list(g.edges())
            if edges:
                edge_to_delete = random.choice(edges)
                g.remove_edge(*edge_to_delete)
                
    return g

#%% create connected subgraphs/query graphs

# Number of nodes in each subgraph
num_nodes_per_subgraph = 7

# Ensure that G has at least num_subgraphs * num_nodes_per_subgraph nodes
if len(g1.nodes) < NumQ * num_nodes_per_subgraph:
    print("The graph does not have enough nodes to generate 100 subgraphs of 7 nodes each.")
else:
    # num = 0
    yes1 = 0
    yes2 = 0
    DFGW = np.zeros(NumQ)
    DIA = []
    
    for num in range(NumQ):
        print("num=",num)
        #%% create connected query graphs by BFS
        if Is_create_query:
            # Randomly select a starting node
            start_node = random.choice(list(g1.nodes()))
    
            # Initialize a subgraph with the starting node
            subgraph = nx.Graph()
            subgraph.add_node(start_node, attr_name = g1.nodes[start_node].get("attr_name",None) )
    
            # Use a breadth-first search (BFS) to add connected nodes to the subgraph
            queue = [start_node]
            while len(subgraph) < num_nodes_per_subgraph and queue:
                current_node = queue.pop(0)
                neighbors = list(g1.neighbors(current_node))
                random.shuffle(neighbors)  # Shuffle neighbors for randomness
                for neighbor in neighbors:
                    if neighbor not in subgraph and len(subgraph) < num_nodes_per_subgraph:
                        subgraph.add_node(neighbor, attr_name = g1.nodes[neighbor].get("attr_name",None) )
                        subgraph.add_edge(current_node, neighbor)
                        queue.append(neighbor)
    
            g2_nodummy = subgraph
        
        #%% add noise to query
        if Is_fea_noise or Is_str_noise:
            g2_nodummy = add_noise_to_query(g2_nodummy, mean_fea = mean_fea, std_fea = std_fea, str_mean= str_mean, str_std= str_std,
                                   Is_fea_noise=Is_fea_noise, Is_str_noise=Is_str_noise)
            
#%%
        N = len(g2_nodummy.nodes)
        # G1 = Graph(g1)
        G2_nodummy = Graph(g2_nodummy)


#%%     only allow the query is connected
        is_connected = nx.is_connected(g2_nodummy)
        if is_connected == 0:
            print("'The query graph is not connected.'")
            continue
        # %% plot the graphs
        # if Is_fig == 1:
        #     vmin = 0
        #     vmax = 9  # the range of color
        #     plt.figure(figsize=(8, 5))
        #     # create some bugs in the nx.draw_networkx, don't know why.
        #     draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
        #     draw_rel(g2_nodummy, vmin=vmin, vmax=vmax,
        #               with_labels=True, shiftx=3, draw=False)
        #     plt.title('Original target graph and query graph: Color indicates the label')
        #     plt.show()
        
        #%% Using the diameter constraint, but the sliding subgraph grows up quickly 
        # def grow_subgraph(graph, center_node, target_diameter):
        #     subgraph = nx.Graph()  # Initialize an empty subgraph
        #     subgraph.add_node(center_node)  # Start with the center node
        #     current_diameter = 0
        
        #     while current_diameter < target_diameter:
        #         neighbors = list(subgraph.nodes())  # Get the current nodes in the subgraph
        #         new_neighbors = []
        
        #         for node in neighbors:
        #             new_neighbors.extend(graph.neighbors(node))
        
        #         # Remove nodes already in the subgraph to avoid duplicates
        #         new_neighbors = set(new_neighbors) - set(subgraph.nodes())
        
        #         # Stop if there are no more new neighbors to add
        #         if not new_neighbors:
        #             break
        
        #         subgraph.add_nodes_from(new_neighbors)  # Add new neighbors to subgraph
        #         current_diameter += 1
        
        #     return subgraph
        
        # def find_subgraph_with_diameter(graph, diameter):
        #     induced_subgraph_list=[]
        #     for center_node in graph.nodes():
        #         subgraph_only_nodes = grow_subgraph(graph, center_node, diameter)
        #         induced_subgraph = graph.subgraph(subgraph_only_nodes.nodes())
        #         # if nx.diameter(induced_subgraph) == diameter:
        #         induced_subgraph_list.append(induced_subgraph)                   
                    
        #     return induced_subgraph_list
        
        
        # g2_diameter = nx.diameter(g2_nodummy)
        # g1_subgraph_list = find_subgraph_with_diameter(g1, diameter=g2_diameter)


        #%% sliding window 
        # diameter of query 
        g2_diameter = nx.diameter(g2_nodummy)
        
        # define a center, return the longest possible length of path from the center node
        def find_center_with_smallest_avg_hops(graph):
            min_avg_hops = float('inf')
            center_node_query = None
            
            for node in graph.nodes():
                avg_hops = sum(nx.shortest_path_length(graph, source=node).values()) / (len(graph.nodes()) - 1)
                
                if avg_hops < min_avg_hops:
                    min_avg_hops = avg_hops
                    center_node_query = node
                    
            longest_path_center = max(nx.shortest_path_length(graph, source=center_node_query).values())
            
            return longest_path_center
        
        g2_longest_path_center = find_center_with_smallest_avg_hops(g2_nodummy)
        
        # Using h-diameter neighborhood hops 
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
        
        #%% go over every node in target
        g1_subgraph_list=[]
        dfgw_sub = []
        transp_FGWD_sub = []
        G1_subgraph_sub = []
        dw_sub = []
        
        ii=0
        for center_node in g1.nodes():
            print(ii)
            ii+=1
        
            # induced_subgraph = create_h_hop_subgraph(g1, center_node, h=math.ceil(g2_diameter/2))  # sometimes could not include the subgraph in the big graph
            # induced_subgraph = create_h_hop_subgraph(g1, center_node, h=math.ceil(g2_diameter))
            g1_subgraph = create_h_hop_subgraph(g1, center_node, h = g2_longest_path_center)
            g1_subgraph_list.append(g1_subgraph)                   

            G1_subgraph = Graph(g1_subgraph)
            
            if len(G1_subgraph.nodes()) < len(G2_nodummy.nodes()):  
                print("The sliding subgraph did not get enough nodes.")
                continue
            
            G2 = copy.deepcopy(G2_nodummy)
            
            if fea_metric == 'jaccard':
                G2.add_attributes({len(G2.nodes()): "0"})  # add dummy            
            else:
                G2.add_attributes({len(G2.nodes()): 0})  # add dummy      
            
    
            # %% plot the graphs
            if Is_fig == 1:
                vmin = 0
                vmax = 9  # the range of color
        
                plt.figure(figsize=(8, 5))
                # create some bugs in the nx.draw_networkx, don't know why.
                draw_rel(g1_subgraph, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
                draw_rel(g2_nodummy, vmin=vmin, vmax=vmax,
                          with_labels=True, shiftx=3, draw=False)
                plt.title('Sliding subgraph and query graph: Color indicates the label')
                plt.show()
    
            # %% weights and feature metric
            p1 = ot.unif(len(G1_subgraph.nodes()))
            # ACTUALLY NOT USED IN THE ALGORITHM
            p2_nodummy = 1/len(G1_subgraph.nodes()) * np.ones([len(G2_nodummy.nodes())])
            p2 = np.append(p2_nodummy, 1-sum(p2_nodummy))
    
            
    
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
            epsilon = 1e-9
            alpha = 0
            dw, log_WD, transp_WD, M, C1, C2  = Fused_Gromov_Wasserstein_distance(
                alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1_subgraph, G2, p1, p2, p2_nodummy)
            
            if dw > epsilon:
                print("filter out")
                continue
            
            dw_sub.append(dw)
            
            #%% FGWD
            alpha = 0.1
            dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
                alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1_subgraph, G2, p1, p2, p2_nodummy)
            
            #%% 
            dfgw_sub.append(dfgw)
            transp_FGWD_sub.append(transp_FGWD)
            G1_subgraph_sub.append(G1_subgraph)
            
        
            
            # %% FGWD, find alpha
            # alld=[]
            # x=np.linspace(0,1,10)
            # for alpha in x:
            #     d,log,transp=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
            #     alld.append(d)
            # fig=plt.figure(figsize=(10,8))
            # plt.plot(x,alld)
            # plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(x[np.argmax(alld)]))
            # plt.xlabel('Alpha')
            # plt.xlabel('FGW dist')
            # plt.show()
    
            # # optimal matching
            # fig=plt.figure(figsize=(10,8))
            # thresh=0.004
            # alpha_opt=x [ alld.index(max(alld)) ]
            # dfgw_opt,log_FGWD_opt,transp_FGWD_opt=Fused_Gromov_Wasserstein_distance(alpha=alpha_opt,features_metric=fea_metric).graph_d(G1,G2,p1,p2,p2_nodummy)
            # # d=dfgw.graph_d(g1,g2)
            # # plt.title('FGW coupling, dist : '+str(np.round(dfgw,3)),fontsize=15)
            # plt.title('FGW coupling, alpha = opt')
            # draw_transp(G1,G2,transp_FGWD_opt,shiftx=2,shifty=0.5,thresh=thresh,
            #             swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
            # plt.show()
    
            # print('Wasserstein distance={}, Gromov distance={} \nFused Gromov-Wasserstein distance for alpha {} = {}'.format(dw,dgw,alpha,dfgw))

        #%% get the min dfgw from the sliding subgraphs
        dgfw_sub_min = min(dfgw_sub)
        min_index = dfgw_sub.index(dgfw_sub_min)
        
        transp_FGWD_sub_min = transp_FGWD_sub[min_index]
        G1_subgraph_min = G1_subgraph_sub[min_index]
        
        dw_sub_min = min(dw_sub)
        
        print("FGWD", dgfw_sub_min)
        print("transp", transp_FGWD_sub_min)
        print("WD", dw_sub_min)

        # if Is_fig == 1:
        vmin = 0
        vmax = 9  # the range of color
        fig = plt.figure(figsize=(10, 8))
        plt.title('Optimal FGWD coupling')
        draw_transp(G1_subgraph_min, G2, transp_FGWD_sub_min, shiftx=2, shifty=0.5, thresh=thresh,
                    swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
        plt.show()
        

    #%% get the final result for one query graph
    dgfw_sub_min_norm=dgfw_sub_min/N # modified obj values 
    DFGW[num] = dgfw_sub_min_norm
    if dgfw_sub_min_norm < thre1:
        yes1 += 1
    if dgfw_sub_min_norm < thre2:
        yes2 += 1
        
    DIA.append(g2_diameter) # for different diameter
            
        
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

# %% rates of all query graphs
print('Rate 1:',yes1/NumQ)
print('Rate 2:',yes2/NumQ)
print('STD:',np.std(DFGW))

DFGW_set.append(DFGW)
Percent1.append(yes1/Num)
Percent2.append(yes2/Num)

Mean.append(np.mean(DFGW))
STD.append(np.std(DFGW))

#create 95% confidence interval for population mean weight
lower, upper = st.norm.interval(confidence=0.95, loc=np.mean(DFGW), scale=st.sem(DFGW))
Lower.append(lower)
Upper.append(upper)

#%% for different diameter
# # Create empty lists for each category
# category_arrays = [[] for _ in range(N)]

# # Iterate through numbers and append to respective category arrays
# for number, category in zip(DFGW, DIA):
#     category_arrays[category - 1].append(number)

# # Print the arrays for each category
# for i, category_array in enumerate(category_arrays):    
#     Mean.append(np.mean(category_array))
#     STD.append(np.std(category_array))
#     Percent1.append(len([num for num in category_array if num < thre1])/len(category_array))
#     Percent2.append(len([num for num in category_array if num < thre2])/len(category_array))
#     #create 95% confidence interval for population mean weight
#     lower, upper = st.norm.interval(confidence=0.95, loc=np.mean(category_array), scale=st.sem(category_array))
#     Lower.append(lower)
#     Upper.append(upper)

# %% boxplot
# fig, ax = plt.subplots()
# # ax.set_title('Hide Outlier Points')
# ax.boxplot(DFGW_set, showfliers=False, showmeans=False)
# # %% plot mean and STD
# plt.figure()
# plt.plot(np.array([0]), np.array(Mean), 'k-+')
# # plt.fill_between(np.array(NN3), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD), alpha=0.5) # alpha here is transparency
# plt.fill_between(np.array([0]), np.array(Lower), np.array(Upper), facecolor = 'k',alpha=0.5) # alpha here is transparency
# plt.grid()
# # plt.xlabel('Size of test graph')
# # plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.ylabel('Mean and 95% confidence interval')
# # %% plot percentage
# plt.figure()
# plt.plot(np.array([0]), np.array(Percent1),'k-x', label='exact match')
# plt.plot(np.array([0]), np.array(Percent2),'k--.', label='approx match')
# plt.grid()
# # plt.xlabel('Size of test graph')
# # plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.ylabel('Success rate')
# plt.legend()
#%% subsitute back the transport matrix
# n1 = len(G1.nodes())
# n2 = len(G2.nodes())
# constC,hC1,hC2=init_matrix(C1,
#                             C2[0:n2-1,0:n2-1],
#                             transp_FGWD[:,0:len(transp_FGWD[0])-1],
#                             p1,
#                             p2_nodummy,
#                             loss_fun='square_loss')
# check_gwloss=gwloss(constC,hC1,hC2,transp_FGWD)
# print(check_gwloss)
# check_wloss=np.sum(transp_FGWD*M)
# print(check_wloss)
# check_fgwloss = (1-alpha)*check_wloss+alpha*check_gwloss
# print(check_fgwloss)
# %% subsitute back the transport matrix
# n1 = len(G1.nodes())
# n2 = len(G2.nodes())
# # constC,hC1,hC2=init_matrix(C1,
# #                             C2[0:n2-1,0:n2-1],
# #                             transp_FGWD[:,0:len(transp_FGWD[0])-1],
# #                             p1,
# #                             p2_nodummy,
# #                             loss_fun='square_loss')
# # check_gwloss=gwloss(constC,hC1,hC2,transp_FGWD)
# check_gwloss=gwloss(cal_L(C1,C2),transp_FGWD)
# print(check_gwloss)
# check_wloss=np.sum(transp_FGWD*M)
# print(check_wloss)
# check_fgwloss = (1-alpha)*check_wloss+alpha*check_gwloss
# print(check_fgwloss)
