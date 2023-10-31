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
# sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib1'))

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
import string
import time

N = 5  # nodes in query
# NN =  [5,10,15,25,35,45,55]
# NN =[10]
# NN = [10]
# N2 = 25 # additional nodes in large graph
# NN2 =[5,10,15,25,35,45,55,65,75,85]
# NN2 =  [75,70,65,55,45,35,25]
# NN2=[5]
# N3 = [x+N for x in NN2]
# NN3 = [15,20,25,35,45,55,65,75,85,95]
# NN3 = [20,50,100,200,300,400,500]
# NN3 = [20,50,100,500,1000,2000,3000]
# NN3 = [50, 100, 1000, 3000, 5000, 7000, 10000]
NN3 = [100, 1000, 3000, 5000, 7000, 10000]
# # NN3 = [10000]
# NN3 = [15,20,25]
# NN3 = [15,45,75]
# N3 = N+N2
# N3 = 45
# NN3 = [45]
# Pw = np.linspace(0.1, 1, 10)
# Pw = np.linspace(0.01, 0.1, 10)
d = 2
# Deg = [0.5,1]+[x for x in range(2, 15, 2)]
# deg = 10
deg = 3
# Pw2 = [deg / (N3-1) for deg in Deg]
# Pw = [0.1]
# pw1 = 0.5 # query
pw1 = d / (N-1)
# pw1 = np.random.choice(np.linspace(0.1, 1, 10))
# pw2 = 0.5 # target
# pw1 = deg / (N-1)
# pw2 = deg / (N3-1)
# Sigma2=[0.01,0.1,0.5,1,2,3,4]
# Sigma2=[0.01]
# sigma1=0.1
# sigma2=0.1
numfea = 100
# NumFea = list(range(1, 20))  # from 1 to 20
# NumFea = [x for x in range(2, 41, 2)]
# NumFea = [2]

# Alpha = np.linspace(0, 1, 11)

# Dia = [i for i in range(1, N)]

stopThr = 1e-9

thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-2
thre3 = 0.05
epsilon = thre1
        
Is_fig = 0
Is_info = 0
Is_fea_noise = 0
Is_str_noise = 0

Num = 1 # number of repeats (generate a random graph and a query)
fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'
loss_fun='square_loss'
# loss_fun = 'kl_loss'

alpha1 = 0
alpha2 = 0.5

mean_fea = 0
std_fea = 0
str_mean = 0
str_std = 0

#%% bootstrap 
def bootstrap_mean_confidence_interval(data, num_bootstraps=1000, alpha=0.05):
    """
    Calculate the 1-alpha confidence interval for the mean using bootstrap resampling.

    Parameters:
    data (numpy array or list): The dataset for which the confidence interval is to be calculated.
    num_bootstraps (int): The number of bootstrap samples to generate.
    alpha (float): The significance level (e.g., 0.05 for a 95% confidence interval).

    Returns:
    tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Initialize an array to store bootstrap means
    bootstrap_means = np.zeros(num_bootstraps)

    # Perform bootstrap resampling and calculate means
    for i in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Sort the bootstrap means
    bootstrap_means.sort()

    # Calculate the lower and upper percentiles for the confidence interval
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return lower_bound, upper_bound

# %% build fully connected graph
def build_fully_graph(N=30, numfea=3):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    g = Graph()
    g.add_nodes(list(range(N)))
    # Fea = np.linspace(0,20,numfea)
    Fea = list(range(0, numfea))
    for i in range(N):
        # g.add_one_attribute(i,v[i])
        # g.add_one_attribute(i,2)
        fea = random.choice(Fea)
        g.add_one_attribute(i, fea)
        for j in range(i+1, N):
            if j != i:
                g.add_edge((i, j))

    return g

# %% build comunity graphs with different assortivity
# pw is the possibility that one edge is connected
def build_comunity_graph(N=30, numfea=3, pw=0.5, fea_metric= 'dirac'):
    g = Graph()
    g.add_nodes(list(range(N)))
    
    if fea_metric == 'dirac' or fea_metric == 'sqeuclidean':
        # v=mu+sigma*np.random.randn(N);
        # v=np.int_(np.floor(v)) # discrete attributes
        # Fea = np.linspace(0,20,numfea)
        Fea = list(range(0, numfea))
        for i in range(N):
            # g.add_one_attribute(i,v[i])
            fea = random.choice(Fea)
            g.add_one_attribute(i, fea)
            for j in range(i+1, N):
                if j != i:
                    r = np.random.rand()
                    if r < pw:
                        g.add_edge((i, j))
                        
    elif fea_metric == 'jaccard':
        for i in range(N):
            # Generate a random length between 1 and 20
            random_length = random.randint(1, 20)
            # Generate a random string of that length
            random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random_length))
            fea = random_string
            g.add_one_attribute(i, fea)
            for j in range(i+1, N):
                if j != i:
                    r = np.random.rand()
                    if r < pw:
                        g.add_edge((i, j))
    
    return g

#%%
def build_line_graph(N=30, numfea=3, fea_metric= 'dirac'):
    g = Graph()
    g.add_nodes(list(range(N)))
    
    if fea_metric == 'dirac':
        # v=mu+sigma*np.random.randn(N);
        # v=np.int_(np.floor(v)) # discrete attributes
        # Fea = np.linspace(0,20,numfea)
        Fea = list(range(0, numfea))
        for i in range(N):
            # g.add_one_attribute(i,v[i])
            fea = random.choice(Fea)
            g.add_one_attribute(i, fea)
        for i in range(N-1):
            g.add_edge((i, i+1))
    
    return g

# %% merge community graphs
def merge_graph(g1, g2):  # inputs are nx_graph
    gprime = nx.Graph(g1)
    N0 = len(gprime.nodes())
    g2relabel = nx.relabel_nodes(g2, lambda x: x + N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True))
    gprime.add_edge(N0-1, N0)

    return gprime

# %% build random graph G1
def build_G1(G, N2=30, numfea=3, pw=0.5, fea_metric= 'dirac'):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    # Fea = np.linspace(0,20,numfea)
    
    L = len(G.nodes())
    # G.add_nodes(list(range(N2)))

    NN = N2+L  # total number of nodes in test graph
    
    if fea_metric == 'dirac' or fea_metric == 'sqeuclidean':
        Fea = list(range(0, numfea))
        for i in range(L, NN):
            # G.add_one_attribute(i,v[i-L])
            fea = random.choice(Fea)
            G.add_one_attribute(i, fea)
            
    elif fea_metric == 'jaccard':
        for i in range(L, NN):
            # Generate a random length between 1 and 20
            random_length = random.randint(1, 20)
            # Generate a random string of that length
            random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random_length))
            fea = random_string
            G.add_one_attribute(i, fea)


    for i in range(NN):
        for j in range(i+1, NN):
            if j != i and j not in range(L):  # no additional edge within the subgraph
            # if j != i and j not in range(L) and j not in G.nx_graph._adj[i].keys():
                r = np.random.rand()  # uniform betweeen [0,1)
                if r < pw:
                    G.add_edge((i, j))

    return G
    
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

for N3 in NN3:
    num = 0

    #%% save data for all iterations
    DFGW = np.zeros(Num)
    transp_FGWD_sliding_min_x = []
    g1_sliding_min_x = []
    G1_sliding_min_x = []
    dw_sliding_x = []
    
    # g1_subgraph_found_x = []
    g2_nodummy_x = []
    # DIA = []
    # yes=0
    index3 = []
    
    time_x= np.zeros(Num)
    
    while num < Num:

        plt.close("all")
        
        print("iter", num)
        
        # %% build a random subgraoh

        # pw1=0.1
        G11 = build_comunity_graph(N=N, numfea=numfea, pw=pw1, fea_metric=fea_metric)
        
        # build line graph for query
        # G11 = build_line_graph(N=N, numfea=numfea, fea_metric=fea_metric)
        
        # %% build G1

        # np.random.seed()  # different graph G1 every time
        G12 = copy.deepcopy(G11)  # initialize with subgraph
        # G111=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G112=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G1 = Graph(merge_graph(G111.nx_graph,G112.nx_graph))
        N2 = N3 - N
        # pw2=pw
        # pw2 = Pw2 [ NN3.index(N3) ]
        pw2 = deg / (N3-1)
        G1 = build_G1(G12, N2=N2, numfea=numfea, pw=pw2, fea_metric=fea_metric)

        # %% G1 is the test graph and G2_nodummy is the query graph
        G2_nodummy = copy.deepcopy(G11)
        # G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)
        
        g1 = G1.nx_graph
        g2_nodummy = G2_nodummy.nx_graph
        
        #%% only allow the query is connected
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
        

        #%% Sliding window: deal with G2_nodummy
        # diameter of query 
        # g2_diameter = nx.diameter(g2_nodummy)
        # DIA.append(g2_diameter) # for different diameter

        # define a center, return the longest possible length of path from the center node
        # def find_center_with_smallest_avg_hops(graph):
        #     min_avg_hops = float('inf')
        #     center_node_query = None
            
        #     for node in graph.nodes():
        #         avg_hops = sum(nx.shortest_path_length(graph, source=node).values()) / (len(graph.nodes()) - 1)
                
        #         if avg_hops < min_avg_hops:
        #             min_avg_hops = avg_hops
        #             center_node_query = node
                    
        #     longest_path_center = max(nx.shortest_path_length(graph, source=center_node_query).values())
            
        #     return longest_path_center
        
        # g2_longest_path_from_center = find_center_with_smallest_avg_hops(g2_nodummy)
        
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
            
            if fea_metric == 'jaccard':
                G2.add_attributes({len(G2.nodes()): "0"})  # add dummy            
            else:
                G2.add_attributes({len(G2.nodes()): 0})  # add dummy      
            time2=time.time()
    
            # %% plot the graphs
            if Is_fig == 1:
                vmin = 0
                vmax = 9  # the range of color
        
                plt.figure(figsize=(8, 5))
                # create some bugs in the nx.draw_networkx, don't know why.
                draw_rel(g1_sliding, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
                draw_rel(g2_nodummy, vmin=vmin, vmax=vmax,
                          with_labels=True, shiftx=3, draw=False)
                plt.title('Sliding subgraph and query graph: Color indicates the label')
                plt.show()
    
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
                alpha=alpha1, features_metric=fea_metric, method=str_metric, loss_fun=loss_fun).graph_d(G1_sliding, G2, p1, p2, p2_nodummy, stopThr=stopThr)
            time4=time.time()
            
            if dw > epsilon:
                print("filter out")
                continue # go to the next sliding subgraph
            time5=time.time()
            
            # %% FGWD
            # alpha = 0.5
            dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
                alpha=alpha2, features_metric=fea_metric, method=str_metric, loss_fun=loss_fun).graph_d(G1_sliding, G2, p1, p2, p2_nodummy, stopThr=stopThr)
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

        
        #%% get the min dfgw from the sliding records
        dgfw_sliding_min = min(dfgw_sliding_list)
        
        min_index = dfgw_sliding_list.index(dgfw_sliding_min)
        
        transp_FGWD_sliding_min = transp_FGWD_sliding_list[min_index]        
        g1_sliding_min = g1_sliding_list[min_index]      
        G1_sliding_min = G1_sliding_list[min_index]      
        dw_sliding_min = dw_sliding_list[min_index]
        
        print("FGWD", dgfw_sliding_min)
        # print("transp", transp_FGWD_sliding_min)
        print("WD", dw_sliding_min)

        if Is_fig:
            vmin = 0
            vmax = 9  # the range of color
            fig = plt.figure(figsize=(10, 8))
            plt.title('Optimal FGWD coupling')
            draw_transp(G1_sliding_min, G2, transp_FGWD_sliding_min, shiftx=2, shifty=0.5, thresh=thresh,  # check the node order when drawing
                        swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
            plt.show()
        

        # %% contruct the found subgraph            
        # # indexes of the largest values of each column except the last column
        # transp_FGWD_sliding_min_nolast = transp_FGWD_sliding_min[:, 0:-1]
        # index = np.argwhere(transp_FGWD_sliding_list_min_nolast == np.max(transp_FGWD_sliding_list_min_nolast,axis=0))      
        # # Get the indices that would sort the second column in ascending order
        # sort_indices = np.argsort(index[:, 1])
        # index = index[sort_indices]
        # nodes_found = set(list((index[:, 0])))
        
        # nodes_found = set(np.argmax(transp_FGWD_sliding_min[:, :-1], axis=0)) # be careful about the node indices
        
        # g1_subgraph_found = g1.subgraph(nodes_found).copy() # taken out from g1 is the induced subgraph that we found 
        
        # def info(g):
        #     # print the features of g
        #     for node in g.nodes(data=True):
        #         node_id, attributes = node
        #         for key, value in attributes.items():
        #             print(f"Node {node_id}:", f"{key}: {value}")
                    
        #     # print the adj matrix of g
        #     g_adj = nx.to_numpy_array(g)
        #     print("Adjacency Matrix:", g_adj)
            
        # if Is_info:
        #     info(g1_subgraph_found)
        #     info(g2_nodummy)
        
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
        
        if check_transp(transp_FGWD_sliding_min, g1_sliding_min, g2_nodummy, Is_info):
            print("These two graphs are the same.")
            # yes+=1
            index3.append(1)
        else:
            index3.append(0)

        # %% keep a record of this iter
        
        DFGW[num] = dgfw_sliding_min # final results of this iter (2 random graphs)
        time_x[num] = sliding_time + time_center
        print("time", time_x[num])
        
        transp_FGWD_sliding_min_x.append(transp_FGWD_sliding_min)        
        g1_sliding_min_x.append(g1_sliding_min)
        G1_sliding_min_x.append(G1_sliding_min)
        dw_sliding_x.append(dw_sliding_min)
        
        # g1_subgraph_found_x.append(g1_subgraph_found)
        g2_nodummy_x.append(g2_nodummy)
        
        #%%
        num +=1 # only succeed then proceed
        
    # %%
    index1 = [index for index, value in enumerate(DFGW) if value < thre1]
    Rate1 = len(index1) / Num
    index2 = [index for index, value in enumerate(DFGW) if value < thre2]
    Rate2 = len(index2) / Num
    index3 = [index for index, value in enumerate(index3) if value == 1]
    # index3 = [i for i in range(len(g2_nodummy_x)) if nx.is_isomorphic(g2_nodummy_x[i], g1_subgraph_found_x[i])]
    Rate3 = len(index3) / Num 
    # Rate3 = yes / Num
    index4 = [index for index, value in enumerate(DFGW) if value < thre3]
    Rate4 = len(index4) / Num
    
    print('Rate 1: FGWD is almost zero', Rate1)
    print('Rate 2: find the approx matching:', Rate2)
    print('Rate 3: the matching is exactly right', Rate3)
    # print('STD:',np.std(DFGW))
    
    DFGW_set.append(DFGW)
    Percent1.append(Rate1)
    Percent2.append(Rate2)
    Percent3.append(Rate3)
    Percent4.append(Rate4) # thre3-Percent4
    
    Percent_set = [Percent1, Percent2, Percent3, Percent4]
    
    Mean.append(np.mean(DFGW))
    STD.append(np.std(DFGW))
    Time.append(np.mean(time_x))
    print(Time)
    #create 95% confidence interval for population mean weight
    # lower, upper = st.norm.interval(confidence=0.95, loc=np.mean(DFGW), scale=st.sem(DFGW))
    lower, upper = bootstrap_mean_confidence_interval(DFGW,alpha=0.05)
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
# %% plot mean and STD
plt.figure()
plt.plot(np.array(NN3), np.array(Mean), 'b-+')
# plt.fill_between(np.array(NumFea), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD), alpha=0.5) # alpha here is transparency
plt.fill_between(np.array(NN3), np.array(Lower), np.array(Upper), facecolor = 'b',alpha=0.5) # NumFea here is transparency
plt.grid()
plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.xlabel('Average node NN3ree of test graph')
# plt.xlabel('Alpha')
plt.ylabel('Mean and 95% confidence interval')
# plt.ylim(0, 0.05)

# %% plot percentage
plt.figure()
plt.plot(np.array(NN3), np.array(Percent1),'-x', color = 'tab:green', label='m='+str(N)+', nFGWD <'+str(thre1))
# plt.plot(np.array(NumFea), np.array(Percent2),'r--.', label='FGWD < ' +str(thre2) +'(approx match)')
plt.plot(np.array(NN3), np.array(Percent4),'--.', color = 'tab:orange', label='m='+str(N)+', nFGWD < ' +str(thre3) )
plt.plot(np.array(NN3), np.array(Percent3),'-+', color = 'b', label='exact matching')
plt.grid()
plt.legend()
plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.xlabel('Average node NN3ree of test graph')
# plt.xlabel('Alpha')
plt.ylabel('Success rate')
# plt.ylim(-0.05, 1.05)

# %% plot time
plt.figure()
plt.plot(np.array(NN3), np.array(Time),'b-+')
plt.grid()
plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
# plt.xlabel('Connectivity of graphs')
# plt.xlabel('Average node degree of test graph')
# plt.xlabel('Alpha')
plt.ylabel('Time (sec)')
# plt.ylim(0, 0.03)
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
