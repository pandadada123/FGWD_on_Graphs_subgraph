# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""


import numpy as np
import os
import sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath(
    'E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
import random
import ot
import networkx as nx
import matplotlib.pyplot as plt
import copy
from ot_distances import Fused_Gromov_Wasserstein_distance
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
# from data_loader import load_local_data,histog,build_noisy_circular_graph
from FGW import init_matrix,gwloss
# from FGW import cal_L,tensor_matrix,gwloss
import scipy.stats as st

N = 10  # nodes in subgraph
# NN =  [5,10,15,25,35,45,55]
# NN =[10]
# NN = [10]
# N2 = 25 # additional nodes in large graph
# NN2 =[5,10,15,25,35,45,55,65,75,85]
# NN2 =  [75,70,65,55,45,35,25]
# NN2=[5]
# N3 = [x+N for x in NN2]
# NN3 = [15,20,25,35,45,55,65,75,85,95]
# N3 = N+N2
N3 = 45
# NN3 = [65]
Pw = np.linspace(0.1, 1, 10)
# pw2 = 0.5
# pw1 = 0.5
# pw2 = 0.5
# Sigma2=[0.01,0.1,0.5,1,2,3,4]
# Sigma2=[0.01]
# sigma1=0.1
# sigma2=0.1
numfea = 4
# NumFea = list(range(1, 11))  # from 1 to 20
# NumFea = [2]

# Alpha = np.linspace(0, 1, 11)

Is_info = 0

DFGW_set = []
Percent1 = []
Percent2 = []
Mean = []
STD = []
Lower = []
Upper = []

# %% build star graph


def build_star_graph():
    g = Graph()
    g.add_attributes({0: 0, 1: 3, 2: 5, 3: 7})    # add color to nodes
    g.add_edge((0, 1))
    g.add_edge((1, 2))
    g.add_edge((1, 3))

    return g

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


def build_comunity_graph(N=30, numfea=3, pw=0.5):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    g = Graph()
    g.add_nodes(list(range(N)))
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


def build_G1(G, N2=30, numfea=3, pw=0.5):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes
    # Fea = np.linspace(0,20,numfea)
    Fea = list(range(0, numfea))

    L = len(G.nodes())
    # G.add_nodes(list(range(N2)))

    NN = N2+L  # total number of nodes in test graph
    for i in range(L, NN):
        # G.add_one_attribute(i,v[i-L])
        fea = random.choice(Fea)
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

for pw in Pw:
    Num = 1000 # number of random graphs
    num = 0
    yes1 = 0
    yes2 = 0
    DFGW = np.zeros(Num)
    while num < Num:

        plt.close("all")

        # %%
        # N=5
        # mu1=-1.5
        # mu2=1.5
        # mu3=3
        # pw1=0.8
        # pw2=0.3
        # pw3=0.8
        # vmin=-3
        # vmax=7
        # np.random.seed(12)
        # G11=build_comunity_graph(N=N,mu=mu1,sigma=0.8,pw=pw1)
        # G12=build_comunity_graph(N=N,mu=mu2,sigma=0.8,pw=pw2)
        # G13=build_comunity_graph(N=N,mu=mu3,sigma=0.8,pw=pw3)
        # com_graph={1:G11,2:G12,3:G13}

        # %% merge
        # n=0
        # Num=[]
        # while n<=3:
        #     num=np.random.randint(1,4) # randomly generate a number within [1,2,3]
        #     Num=np.append(Num,num)
        #     g1=merge_graph(g1, com_graph[num].nx_graph)
        #     n+=1

        # G1=Graph(g1)

        # %% build a fully connected graph (also the subgraph)
        # G11=build_fully_graph(N=N,mu=mu1,sigma=0.01)
        # G11 = build_star_graph()
        # G11=build_comunity_graph(N=N,mu=mu1,sigma=2, pw=0.5)

        # %% build a random subgraoh
        # G0 = Graph() # an empty graph
        # different graph with different seed -> same subgraph everytime
        # np.random.seed(12)
        # G11 = build_G1(G0, N=N, numfea = numfea, pw = pw1) # if set pw = 1 to build a fully-conn graph
        # if set pw = 1 to build a fully-conn graph
        pw1=pw
        G11 = build_comunity_graph(N=N, numfea=numfea, pw=pw1)

        # %% build G1
        # np.random.seed()  # different graph G1 every time
        G12 = copy.deepcopy(G11)  # initialize with subgraph
        # G111=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G112=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G1 = Graph(merge_graph(G111.nx_graph,G112.nx_graph))
        N2 = N3 - N
        pw2=pw
        G1 = build_G1(G12, N2=N2, numfea=numfea, pw=pw2)

        # check if all nodes in G1 are connected
        # temp=G1.nx_graph._adj
        # if any(value=={} for value in temp.values()) == True:
        #     print("oops")
        #     continue

        # %%
        G2_nodummy = copy.deepcopy(G11)
        # G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)
        G2 = copy.deepcopy(G2_nodummy)
        G2.add_attributes({len(G2.nodes()): 0})  # add dummy

        # %%  The followings are fixed
        g1 = G1.nx_graph
        g2 = G2.nx_graph
        g2_nodummy = G2_nodummy.nx_graph

        # %% check if every pair of nodes have path
        # n1 = len(G1.nodes())
        # try:
        #     for ii in range(n1):
        #           nx.shortest_path_length(g1,source=0,target=ii)
        # except:
        #     print("oops2")
        #     continue

        # %%
        vmin = 0
        vmax = 9  # the range of color

        # plt.figure(figsize=(8, 5))
        # create some bugs in the nx.draw_networkx, don't know why.
        # draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
        # draw_rel(g2, vmin=vmin, vmax=vmax,
        #          with_labels=True, shiftx=3, draw=False)
        # plt.title('Two graphs. Color indicates the label')
        # plt.show()

        # %% weights and feature metric
        p1 = ot.unif(len(G1.nodes()))
        # ACTUALLY NOT USED IN THE ALGORITHM
        p2_nodummy = 1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])
        p2 = np.append(p2_nodummy, 1-sum(p2_nodummy))

        fea_metric = 'dirac'
        # fea_metric = 'hamming'
        # fea_metric = 'sqeuclidean'
        # str_metrc = 'shortest_path'
        str_metric = 'adj'

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
        alpha = 0.5
        dfgw, log_FGWD, transp_FGWD, M, C1, C2 = Fused_Gromov_Wasserstein_distance(
            alpha=alpha, features_metric=fea_metric, method=str_metric, loss_fun='square_loss').graph_d(G1, G2, p1, p2, p2_nodummy)
        
        # fig = plt.figure(figsize=(10, 8))
        # plt.title('FGWD coupling')
        # draw_transp(G1, G2, transp_FGWD, shiftx=2, shifty=0.5, thresh=thresh,
        #             swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
        # plt.show()

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
        thre1 = 1e-5
        # thre2=-0.015000 # entropic
        thre2 = 0.001
        
        DFGW[num] = dfgw
        if dfgw < thre1:
            yes1 += 1
        if dfgw < thre2:
            yes2 += 1

        num += 1
        print(num)

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

        # %%
    print('Rate 1:',yes1/Num)
    print('Rate 2:',yes2/Num)
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
    
# %% boxplot
fig, ax = plt.subplots()
# ax.set_title('Hide Outlier Points')
ax.boxplot(DFGW_set, showfliers=False, showmeans=False)
# %% plot mean and STD
plt.figure()
plt.plot(np.array(Pw), np.array(Mean), 'k-+')
# plt.fill_between(np.array(Pw), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD), alpha=0.5) # alpha here is transparency
plt.fill_between(np.array(Pw), np.array(Lower), np.array(Upper), facecolor = 'k',alpha=0.5) # alpha here is transparency
plt.grid()
# plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
plt.xlabel('Connectivity of graphs')
plt.ylabel('Mean and STD')
# %% plot percentage
plt.figure()
plt.plot(np.array(Pw), np.array(Percent1),'k-x', label='exact match')
plt.plot(np.array(Pw), np.array(Percent2),'k--.', label='approx match')
plt.grid()
# plt.xlabel('Size of test graph')
# plt.xlabel('Number of features')
plt.xlabel('Connectivity of graphs')
plt.ylabel('Success rate')
plt.legend()
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
