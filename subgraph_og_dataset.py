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
# sys.path.append(os.path.realpath(‘E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_0.0'))

from lib1.graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
import random
import ot
import networkx as nx
import matplotlib.pyplot as plt
import copy
from lib1.ot_distances import Fused_Gromov_Wasserstein_distance
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from lib1.data_loader import load_local_data,histog,build_noisy_circular_graph
# from FGW import init_matrix,gwloss  # lib 0.0 no need
# from FGW import cal_L,tensor_matrix,gwloss
import scipy.stats as st
import time
import pickle


N=6

thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-2
thre3 = 0.05
epsilon = thre1
        
Is_fig = 0
Is_info = 0
Is_create_query = 0
Is_create_query_random = 0
Is_create_query_deter = 0

Is_check_found_subgraph = 0
Is_check_transp = 1

Is_fea_noise = 1
Is_str_noise = 0

Num = 1000 # number of repeats (generate a random graph and a query)
# fea_metric = 'dirac'
# fea_metric = 'hamming'
fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'

alpha = 0.5

mean_fea = 1 # number of nodes that has been changed
std_fea = 0.5 # zero mean Gaussian
str_mean = 0
str_std = 0.1

# %%
DFGW_set = []
Percent1 = []
Percent2 = []
Percent3 = []
Percent4 = []
Mean = []
Time = []
STD = []
Lower = []
Upper = []

# dataset_n='mutag' 
# dataset_n='protein' 
# dataset_n='protein_notfull' 
# dataset_n='aids'
# dataset_n='ptc'   # color is not a tuple
# dataset_n='cox2'
dataset_n='bzr'

# path='/home/pan/dataset/data/'
path='E:/Master Thesis/dataset/data/'
# X is consisted of graph objects
X,label=load_local_data(path,dataset_n,wl=0) # using the "wl" option that computes the Weisfeler-Lehman features for each nodes as shown is the notebook wl_labeling.ipynb
# we do not use WL labeling 
# x=X[2]

plt.close("all")

NumG = len(X) # Number of graphs
NumQ = NumG # Number of query graphs
# NumQ = 1


# num = 0
yes1 = 0
yes2 = 0
yes3 = 0
yes4 = 0
DFGW = np.zeros(NumQ)
time_x= np.zeros(NumQ)
# DIA = []
Ratio = np.zeros(NumQ)
missing_files_count = 0
index3 = []
Ratio = np.zeros(NumQ)


for num in range(NumQ):
    print("num=",num)
    
    #%% create connected query graphs by BFS
    if Is_create_query:
        if Is_create_query_random:
            # Randomly select a graph
            graph_number = random.randint(0, NumG - 1) # randomly select a graph
            # g3 = random.choice(X).nx_graph
            g3 = X[graph_number].nx_graph
            while N > len(g3.nodes()):
                print("The required query is too big")
                g3 = random.choice(X).nx_graph
    
            # Randomly select a starting node
            start_node = random.choice(list(g3.nodes()))
    
            # Initialize a subgraph with the starting node
            subgraph = nx.Graph()
            subgraph.add_node(start_node, attr_name = g3.nodes[start_node].get("attr_name",None) )
    
            # Use a breadth-first search (BFS) to add connected nodes to the subgraph
            queue = [start_node]
            while len(subgraph) < N and queue:
                current_node = queue.pop(0)
                neighbors = list(g3.neighbors(current_node))
                random.shuffle(neighbors)  # Shuffle neighbors for randomness
                for neighbor in neighbors:
                    if neighbor not in subgraph and len(subgraph) < N:
                        subgraph.add_node(neighbor, attr_name = g3.nodes[neighbor].get("attr_name",None) )
                        subgraph.add_edge(current_node, neighbor)
                        queue.append(neighbor)
    
            g2_nodummy = nx.Graph(subgraph)
            g2_nodummy_orignal = nx.Graph(subgraph) # two g2 are the same
        
        if Is_create_query_deter:
            # Randomly select a graph
            graph_number = random.randint(0, NumG - 1) # randomly select a graph
            # g3 = random.choice(X).nx_graph
            g3 = X[graph_number].nx_graph
            Keys = sorted(list(g3.nodes.keys()))  # [list] order of nodes
            selected_nodes = [node for node in g3.nodes() if node in Keys[:N]]
            subgraph = g3.subgraph(selected_nodes).copy()
            
            g2_nodummy = nx.Graph(subgraph)
            g2_nodummy_orignal = nx.Graph(subgraph) # two g2 are the same
            
        #%% add noise to query
        # in this case, g2_nodummy is the noisy version
        if Is_fea_noise or Is_str_noise:
            g2_nodummy = add_noise_to_query(g2_nodummy, mean_fea = mean_fea, std_fea = std_fea, str_mean= str_mean, str_std= str_std,
                                   Is_fea_noise=Is_fea_noise, Is_str_noise=Is_str_noise)
    
    #%% import query
    else:
        graph_number = num
        
        # Construct the file path
        file_name = str(num) + '.pickle'
        folder_path_1 = "E:\Master Thesis\dataset\data\BZR\query_noise_fea_0_0"
        file_path_1 = os.path.join(folder_path_1, file_name)
    
        # Check if the file exists
        if os.path.exists(file_path_1) == 0:
            # File does not exist
            print(f"File {file_name} does not exist.")
            missing_files_count += 1# Load the pickle file
            
        with open(file_path_1, 'rb') as file1:
                g2_nodummy_original = pickle.load(file1)
        
        folder_path_2 = "E:\Master Thesis\dataset\data\BZR\query_noise_fea_"+str(mean_fea)+"_"+str(std_fea)
        file_path_2 = os.path.join(folder_path_2, file_name)
        with open(file_path_2, 'rb') as file2:
                g2_nodummy = pickle.load(file2)
    

    #%%
    N = len(g2_nodummy.nodes)
    # G1 = Graph(g1)
    G2_nodummy = Graph(g2_nodummy)
        
    #%% only allow the query is connected
    is_connected = nx.is_connected(g2_nodummy)
    if is_connected == 0:
        print("'The query graph is not connected.'")
        continue
     
    #%%
    # if nx.diameter(g2_nodummy) > 2:
    #     print("The diameter is too large")
    #     continue
    # %%
    start_time = time.time()
    
    G2 = copy.deepcopy(G2_nodummy)
    
    Large = 1e6
    if fea_metric == 'jaccard':
        G2.add_attributes({Large: "0"})  # add dummy  
    elif fea_metric == 'sqeuclidean':
        G2.add_attributes({Large: np.array([0])  })  # add dummy 
    elif fea_metric == 'dirac':
        G2.add_attributes({Large: 0})  # add dummy       
        
    g2 = G2.nx_graph

    #%% 
    G1 = X[num]
    g1=G1.nx_graph
    
    # %% check if every pair of nodes have path
    # n1 = len(G1.nodes())
    # try:
    #     for ii in range(n1):
    #           nx.shortest_path_length(g1,source=0,target=ii)
    # except:
    #     print("oops2")
    #     continue

    # %%
    # vmin = 0
    # vmax = 9  # the range of color
    # plt.figure(figsize=(8, 5))
    # create some bugs in the nx.draw_networkx, don't know why.
    # draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
    # draw_rel(g2, vmin=vmin, vmax=vmax,
    #           with_labels=True, shiftx=3, draw=False)
    # plt.title('Two graphs. Color indicates the label')
    # plt.show()

    # %% weights and feature metric
    p1 = ot.unif(len(G1.nodes()))
    # ACTUALLY NOT USED IN THE ALGORITHM
    p2_nodummy = 1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])
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
    
    # FGWD
    # alpha = 0.5
    dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
        alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy)
    
    end_time = time.time()
    
    if Is_fig:
        vmin = 0
        vmax = 9  # the range of color
        fig = plt.figure(figsize=(10, 8))
        plt.title('FGWD coupling')
        draw_transp(G1, G2, transp_FGWD, shiftx=2, shifty=0.5, thresh=thresh,
                    swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
        plt.show()

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

    # %%
    # dfgw=dfgw/N # modified obj values 
    
    DFGW[num] = dfgw
    if dfgw < thre1:
        yes1 += 1
    if dfgw < thre2:
        yes2 += 1
    if dfgw < thre3:
        yes3 += 1
        
    time_x[num] = end_time - start_time
    print("time", time_x[num])
        
    #%% check feature and structure to decide if it find an exact matching
    def check_transp(transp, h1, h2, Is_info):
           
        transp_nolast = transp[:, 0:-1]
        index = np.argwhere(transp_nolast == np.max(transp_nolast,axis=0))     
        # index = np.argwhere(transp_FGWD_sliding_min[:, 0:-1] > 1e-3)
        # Get the indices that would sort the second column in ascending order
        sort_indices = np.argsort(index[:, 1])
        index = index[sort_indices] # sorted with the second column
        
        # feature
        nodes1=h1.nodes() # [dict]
        nodes2=h2.nodes()
        Keys1 = sorted(list(h1.nodes.keys()))  # [list] order of nodes for cost matrices, from small to large
        Keys2 = sorted(list(h2.nodes.keys()))
        Fea1 = []
        Fea2 = []
        for i in range(index.shape[0]):                
            key1 = Keys1[index[i,0]]
            key2 = Keys2[index[i,1]]
            f1 = nodes1[key1]['attr_name']
            f2 = nodes2[key2]['attr_name']
            Fea1.append(f1)
            Fea2.append(f2)
            
        if Fea1 != Fea2:
            print("feature is different")
            return False 
        
        if Is_info: 
            # with ascending order of both graphs
            print("Features of subgraph within the source graph:")
            print(Fea1)
            print("Features of the query graph:")
            print(Fea2)
            
        # structure (create adj matrix in ascending order)
        # print("Neighbours of source subgraph:")
        # Structure_keys = list(h1._node.keys())
        # Structure_source = list(h1._adj.values())
        # Structure_source2 = {}  # the subgraph within the large graph, but with irrelevant nodes
        # for source in index[:, 0]:
        #     Structure_source2[Structure_keys[source]
        #                       ] = Structure_source[source]

        # temp_keys = list(Structure_source2.keys())
        # for key in temp_keys:
        #     for k in Structure_source2[key].copy():
        #         if k not in temp_keys:
        #             # delete the irrelevant nodes
        #             Structure_source2[key].pop(k, None)
        #     print(Structure_source2[key])

        # print("Neighbours of query graph:")
        # Structure_target = list(g2_nodummy._adj.values())
        # for target in index[:, 1]:
        #     print(Structure_target[target])


        # # Adj matrix
        # def generate_adjacency_matrix(graph_dict):
        #     # Get all unique nodes from the dictionary keys
        #     nodes = list(graph_dict.keys())
        #     num_nodes = len(nodes)

        #     # Initialize an empty adjacency matrix with zeros
        #     adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        #     # Iterate over the graph dictionary
        #     for node, connections in graph_dict.items():
        #         # Get the index of the current node
        #         node_index = nodes.index(node)

        #         # Iterate over the connected nodes
        #         for connected_node in connections.keys():
        #             # Get the index of the connected node
        #             connected_node_index = nodes.index(connected_node)

        #             # Set the corresponding entry in the adjacency matrix to 1
        #             adjacency_matrix[node_index][connected_node_index] = 1

        #     return adjacency_matrix

        # adjacency_subgraph = generate_adjacency_matrix(Structure_source2)
        # adjacency_query = generate_adjacency_matrix(g2_nodummy._adj)
        
        
            
        # structure
        A1 = nx.to_numpy_array(h1, nodelist=Keys1)
        A2 = nx.to_numpy_array(h2, nodelist=Keys2)
        
        # Create a submatrix using the index_vector
        a1 = A1[np.ix_(index[:,0], index[:,0])]
        
        # Ensure that the submatrix is symmetric
        a1 = np.maximum(a1, a1.T)
        
        if Is_info: 
            print("Adjacency matrix within the source graph")
            print(a1)

            print("Adjacency matrix of query graph")
            print(A2)

        if np.array_equal(a1, A2) == 0:
            print("structure is different")
            return False
        
        return True
    
    def check_transp2(transp):
         # n = len(transp)
         m = len(transp[0])  # Assuming all rows have the same number of columns
         count_satisfied = 0        
         for i in range(m - 1):  # Iterate through the first (m-1) rows
             row = transp[i]        
             # Check if the ith entry of the ith row is the maximum entry in that row
             if row[i] == max(row):
                 count_satisfied += 1
         # Calculate the ratio
         ratio = count_satisfied / (m - 1) if m > 1 else 0.0
         return ratio
     
    if Is_check_found_subgraph:
        if check_transp(transp_FGWD, g1, g2_nodummy, Is_info):
            print("These two graphs are the same.")
            # yes+=1
            index3.append(1)
        else:
            index3.append(0)
        
    elif Is_check_transp:
        # if graph_number == min_index_x:
        try:
            ratio = check_transp2(transp_FGWD)
            print("ratio", ratio)
            Ratio[num]=ratio 
        # else:
        #     Ratio[num]=0
        except:
            Ratio[num] = 0
            
    # %%
print('Rate 1:',yes1/NumQ)
print('Rate 2:',yes2/NumQ)
print('STD:',np.std(DFGW))

# index3 = [index for index, value in enumerate(index3) if value == 1]
# Rate3= len(index3) / NumQ
# print('Rate 3: the matching is exactly right', Rate3)

if Is_check_found_subgraph: 
    print('Rate: the matching is exactly right', yes3/NumQ)
elif Is_check_transp:
    print('Rate: the ratio of correct nodes', np.mean(Ratio))
    
DFGW_set.append(DFGW)
Percent1.append(yes1/Num)
Percent2.append(yes2/Num) # suboptimal for 0.01
# Percent3.append(Rate3)  # matching is exactly right
Percent4.append(yes3/Num) # suboptimal for 0.05

Mean.append(np.mean(DFGW))
STD.append(np.std(DFGW))
Time.append(np.mean(time_x))
print("average time", Time[0])

print("mean", np.mean(DFGW))
print("std", np.std(DFGW))

#create 95% confidence interval for population mean weight
# lower, upper = st.norm.interval(confidence=0.95, loc=np.mean(DFGW), scale=st.sem(DFGW))
# lower, upper = bootstrap_mean_confidence_interval(DFGW,alpha=0.05)
# Lower.append(lower)
# Upper.append(upper)
    
# %% boxplot
# fig, ax = plt.subplots()
# # ax.set_title('Hide Outlier Points')
# ax.boxplot(DFGW_set, showfliers=False, showmeans=False)
# # %% plot mean and STD
# plt.figure()
# plt.plot(np.array(Alpha), np.array(Mean), 'k-+')
# # plt.fill_between(np.array(Alpha), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD), alpha=0.5) # alpha here is transparency
# plt.fill_between(np.array(Alpha), np.array(Lower), np.array(Upper), facecolor = 'k',alpha=0.5) # alpha here is transparency
# plt.grid()
# # plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# # plt.xlabel('Connectivity of graphs')
# # plt.xlabel('Average node Alpharee of test graph')
# # plt.xlabel('Alpha')
# plt.ylabel('Mean and 95% confidence interval')
# plt.ylim(0, 0.05)

# # %% plot percentage
# plt.figure()
# plt.plot(np.array(Alpha), np.array(Percent1),'r-.x', label='nFGWD <'+str(thre1))
# # plt.plot(np.array(Alpha), np.array(Percent2),'r--.', label='FGWD < ' +str(thre2) +'(approx match)')
# plt.plot(np.array(Alpha), np.array(Percent4),'--.', color = 'tab:blue', label='nFGWD < ' +str(thre3))
# plt.plot(np.array(Alpha), np.array(Percent3),'k-+', label='exact matching')
# plt.grid()
# # plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# # plt.xlabel('Connectivity of graphs')
# # plt.xlabel('Average node Alpharee of test graph')
# # plt.xlabel('Alpha')
# plt.ylabel('Success rate')
# plt.legend()
# plt.ylim(-0.05, 1.05)

# # %% plot time
# plt.figure()
# plt.plot(np.array(Alpha), np.array(Time),'k-x')
# plt.grid()
# # plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# # plt.xlabel('Connectivity of graphs')
# # plt.xlabel('Average node Alpharee of test graph')
# # plt.xlabel('Alpha')
# plt.ylabel('Time (sec)')
# plt.ylim(0, 0.01)

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
