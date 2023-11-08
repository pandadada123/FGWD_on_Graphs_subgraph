# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""


import sys
sys.modules[__name__].__dict__.clear()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

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

# import pickle
import dill as pickle
import time 



N=6

stopThr = 1e-09

thre1 = stopThr
# thre2=-0.015000 # entropic
thre2 = 1e-2
# thre3 = 0.05
# epsilon = thre1
        
Is_fig = 0
Is_info = 0
Is_create_query = 0
Is_create_query_random = 0
Is_create_query_deter = 0

Is_check_found_subgraph = 0
Is_check_transp = 1 # check the ratio of correct nodes

Is_fea_noise = 0
Is_str_noise = 0

Num = 1 # number of repeats (generate a random graph and a query)
# fea_metric = 'dirac'
# fea_metric = 'hamming'
fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'

alpha = 0.5

mean_fea = 1 # number of nodes that has been changed
std_fea = 0.1 # zero mean Gaussian
# str_mean = 0
# str_std = 0.1
NumQ_for_each_graph = 10


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
# dataset_n='bzr'
dataset_n ='firstmm'

# dataset_name = 'BZR'
dataset_name = 'FIRSTMM_DB'

# path='/home/pan/dataset/data/'
path='E:/Master Thesis/dataset/data/'
# X is consisted of graph objects
X,label=load_local_data(path,dataset_n,wl=0) # using the "wl" option that computes the Weisfeler-Lehman features for each nodes as shown is the notebook wl_labeling.ipynb
# we do not use WL labeling 
# x=X[2]

plt.close("all")

NumG = len(X) # Number of graphs
# NumQ = NumG # Number of query graphs
# NumQ = 1

#%% add noise to the query
def add_noise_to_query(g,mean_fea,std_fea,str_mean,str_std,
                       Is_fea_noise,Is_str_noise):    
    if Is_fea_noise: # Add label noise
        selected_nodes = random.sample(g.nodes(), mean_fea)
        
        if fea_metric == 'jaccard':
            for node in selected_nodes:
                current_string = g.nodes[node]['attr_name']
                # Convert the input string to a list of Unicode code points
                code_points = [ord(char) for char in current_string]
            
                # Apply Gaussian noise to each code point
                noisy_code_points = [
                    int(round(code + np.random.normal(0, std_fea)))
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
            for node in selected_nodes:
                current_value = g.nodes[node]['attr_name']
                noise = np.random.normal(0, std_fea)
                new_value = current_value + noise
                g.nodes[node]['attr_name'] = round(new_value)  # still int value
        
        elif fea_metric == 'sqeuclidean': # real value
            for node in selected_nodes:
                current_value = g.nodes[node]['attr_name']
                noise = [np.random.normal(0, std_fea) for _ in range(len(current_value))]
                new_value = [x + y for x, y in zip(current_value, noise)]
                g.nodes[node]['attr_name'] = new_value  
                
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

# num = 0
yes1 = 0
yes2 = 0
yes3 = 0
yes4 = 0
# DFGW = np.zeros(NumG)
# time_x= np.zeros(NumG)
# Ratio = np.zeros(NumG)
DFGW = np.zeros((NumG,NumQ_for_each_graph))
time_x= np.zeros((NumG,NumQ_for_each_graph))
Ratio = np.zeros((NumG,NumQ_for_each_graph))
Index3 = np.zeros((NumG,NumQ_for_each_graph))

missing_files_count = 0
Num_not_connected = 0

for num in range(NumG):
    print("num=", num)
    
    #%% 
    G1 = X[num]
    g1=G1.nx_graph
    
    for numq in range(NumQ_for_each_graph):
        print("numq=", numq)
    
    #%% import query
    
        graph_number = num
        
        # Construct the file path
        file_name = str(num) + '_' + str(numq) + '.pickle'
        # file_name = str(num) + '.pickle'
        folder_path_1 = "E:\\Master Thesis\\dataset\\data\\"+dataset_name+"\\query_noise_fea_0_0"
        file_path_1 = os.path.join(folder_path_1, file_name)
    
        # Check if the file exists
        if os.path.exists(file_path_1) == 0:
            # File does not exist
            print(f"File {file_name} does not exist.")
            missing_files_count += 1# Load the pickle file
            
        with open(file_path_1, 'rb') as file1:
                g2_nodummy_original = pickle.load(file1)
        
        folder_path_2 = "E:\\Master Thesis\\dataset\\data\\"+dataset_name+"\\query_noise_fea_"+str(mean_fea)+"_"+str(std_fea)
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
            
            Num_not_connected += 1
            
            # DFGW[num] = np.nan
            # time_x[num] = np.nan
            # Ratio[num] = np.nan
            DFGW[num,numq] = np.nan
            time_x[num,numq] = np.nan
            Ratio[num,numq] = np.nan
            
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
            alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy, stopThr=stopThr)
        
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
        
        # DFGW[num] = dfgw
        DFGW[num,numq] = dfgw
        
        if dfgw < thre1:
            yes1 += 1
        if dfgw < thre2:
            yes2 += 1
            
        # time_x[num] = end_time - start_time
        time_x[num,numq] = end_time - start_time
        print("time", time_x[num,numq])
            
        #%% check feature and structure to decide if it find an exact matching
        def check_transp(transp, h1, h2, Is_info): # h1 is the big graph, h2 is the subgraph 
               
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
             M = len(transp[0])  # Assuming all rows have the same number of columns
             count_satisfied = 0        
             for i in range(M - 1):  # Iterate through the first (m-1) rows
                 row = transp[i]        
                 # Check if the ith entry of the ith row is the maximum entry in that row
                 if row[i] == max(row):
                     count_satisfied += 1
             # Calculate the ratio
             ratio = count_satisfied / (M - 1) if M > 1 else 0.0
             return ratio
        
        def check_transp3(transp, ground_truth, node_indices_g1_subgraph_min): # subgraph are created randomly
            # Get the number of columns
            M = len(transp[0])
            
            # Ensure the matrix has the expected number of columns
            if M-1 != len(ground_truth):
                raise ValueError("Matrix column count and ground truth length mismatch.")
            
            # Count the number of matches
            matches = sum(node_indices_g1_subgraph_min[np.argmax(transp[:, i])] == ground_truth[i] for i in range(M-1))
            
            # Compute the ratio
            ratio = matches / (M-1)
            
            return ratio
        
        #%% check if it finds the exact matching
        # use the clean graph to check!
        if check_transp(transp_FGWD, g1, g2_nodummy_original, Is_info):
            print("These two graphs are the same.")
            Index3[num,numq]=1
            yes3+=1
        else:
            Index3[num,numq]=0
        
        #%% calculate the ratio 
        # have to know the node indices (do not need to ues the clean graph)
        ground_truth = sorted(list(g2_nodummy.nodes.keys()))
        node_indices_g1 = sorted(list(g1.nodes.keys()))
        try:
            # ratio = check_transp2(transp_FGWD_sub_min)
            ratio = check_transp3(transp_FGWD,ground_truth,node_indices_g1)            
            print("ratio", ratio)
            # Ratio[num]=ratio 
            Ratio[num,numq]=ratio 
        except:
            # Ratio[num] = 0
            Ratio[num,numq] = 0
            
# %%
print('Overall')
Effective_Num = NumG * NumQ_for_each_graph - Num_not_connected
print('Rate 1: FGWD is almost zero', yes1/Effective_Num)
print('Rate 2: find the approx matching:',yes2/Effective_Num)
# print('STD:',np.std(DFGW))
# index3 = [index for index, value in enumerate(index3) if value == 1]
# Rate3= len(index3) / NumQ
print('Rate 3: the matching is exactly right', yes3/Effective_Num)
print('Rate 4: the ratio of correct nodes', np.nanmean(Ratio))
    
DFGW_set.append(DFGW)
# Percent1.append(yes1/Num)
# Percent2.append(yes2/Num) # suboptimal for 0.01
# # Percent3.append(Rate3)  # matching is exactly right
# Percent4.append(yes3/Num) # suboptimal for 0.05

# Mean.append(np.nanmean(DFGW))
# STD.append(np.nanstd(DFGW))
print('average time', np.nanmean(time_x))
print('Num_not_connected:', Num_not_connected)
print("number of nan in DFGW:", np.isnan(DFGW).sum())
print("mean of dfgw", np.nanmean(DFGW))
print("std of dfgw", np.nanstd(DFGW))

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
