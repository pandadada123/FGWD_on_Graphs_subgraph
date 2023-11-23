# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:24:22 2023

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
# from data_loader import load_local_data,histog,build_noisy_circular_graph
# from FGW import init_matrix,gwloss  # lib 0.0 no need
# from FGW import cal_L,tensor_matrix,gwloss
import scipy.stats as st
import time

stopThr = 1e-09

N = 8 # size of query
S = 6 # size of subgraph
N3 = 35
deg = 10
numfea = 15


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

#%% query 
# G11=Graph() # community 1
# G11.add_attributes({0:0,1:1,2:2,3:3,4:4,5:4,6:4})    # add color to nodes
# G11.add_edge((0,1))
# G11.add_edge((0,2))
# G11.add_edge((0,3))
# G11.add_edge((1,2))
# G11.add_edge((1,3))
# G11.add_edge((2,3))

# G11.add_edge((3,4))
# G11.add_edge((4,5))
# G11.add_edge((4,6))
# G11.add_edge((5,6))

#%% subgraph
# G12=Graph() # community 1
# G12.add_attributes({0:0,1:1,2:2,3:3,4:4,5:4,6:4,7:4})    # add color to nodes
# G12.add_edge((0,1))
# G12.add_edge((0,2))
# G12.add_edge((0,3))
# G12.add_edge((1,2))
# G12.add_edge((1,3))
# G12.add_edge((2,3))

# G12.add_edge((3,4))
# G12.add_edge((4,5))
# G12.add_edge((4,6))
# G12.add_edge((4,7))
# G12.add_edge((5,6))
# G12.add_edge((5,7))
# G12.add_edge((6,7))

#%% query
G11 = build_fully_graph(N=N, numfea=2)
    
#%% subgraph
G12 = build_fully_graph(N=S, numfea=2)

#%% test
G13 = copy.deepcopy(G12)  # initialize with subgraph
# G111=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
# G112=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
# G1 = Graph(merge_graph(G111.nx_graph,G112.nx_graph))
N2 = N3 - S
# pw2=pw1
# pw2 = Pw[ NN3.index(N3) ]
pw2 = deg / (N3-1)
G1 = build_G1(G13, N2=N2, numfea=numfea, pw=pw2)

#%%
G2_nodummy = copy.deepcopy(G11)
# G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)        
g1 = G1.nx_graph
g2_nodummy = G2_nodummy.nx_graph

#%% 
vmin = 0
vmax = 9  # the range of color

plt.figure()
# create some bugs in the nx.draw_networkx, don't know why.
draw_rel(g1, vmin=vmin, vmax=vmax, with_labels=True, draw=False)
draw_rel(g2_nodummy, vmin=vmin, vmax=vmax,
          with_labels=True, shiftx=3, draw=False)
plt.title('test graph and query graph: Color indicates the label')
plt.show()


#%% OT-sliding 
gamma = 1
epsilon = 0.2

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

alpha1 = 0
alpha2 = 0.5

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'
# fea_metric = 'jaccard'
# str_metric = 'shortest_path'  # remember to change lib0 and cost matrix
str_metric = 'adj'
loss_fun='square_loss'
# loss_fun = 'kl_loss'

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
    g1_sliding = create_h_hop_subgraph(g1, center_node, h = g2_longest_path_from_center + gamma)
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
    p2_nodummy = 1/len(G1_sliding.nodes()) * np.ones([len(G2_nodummy.nodes())]) * S/N
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
        print(dw)
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

vmin = 0
vmax = 9  # the range of color
fig = plt.figure()
plt.title('Optimal FGWD coupling')
draw_transp(G1_sliding_min, G2, transp_FGWD_sliding_min, shiftx=2, shifty=0.5, thresh=thresh,  # check the node order when drawing
            swipy=True, swipx=False, with_labels=True, vmin=vmin, vmax=vmax)
plt.show()

        