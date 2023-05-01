# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:06:16 2023

@author: Pandadada
"""

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_subgraph/lib_1.0'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
# from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from ot_distances import Fused_Gromov_Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx
import ot

N = 5 # nodes in subgraph
N2 = 20 # additional nodes in large graph
# NN2 =[5,10,20,40,60]
# NN2 =[20]
# Pw1 =  [0.3, 0.5, 0.7, 0.9,1] 
# Pw2 = [0.1, 0.3, 0.5, 0.7, 0.9,1]
pw2=0.5
pw1=0.5
# pw2=0.5
# Sigma2=[0.01,0.1,0.5,1,2,3,4] 
# Sigma2=[0.01]
sigma1=0.1
sigma2=0.1
# Alpha = np.linspace(0, 1, 11)

# N_dum = 1
NN_dum = [1,2,3,4,5]

DFGW_set = []
Percent=[]
Mean=[]
STD=[]

#%% build star graph
def build_star_graph():
    g=Graph()
    g.add_attributes({0:0,1:3,2:5,3:7})    # add color to nodes
    g.add_edge((0,1))
    g.add_edge((1,2))
    g.add_edge((1,3))
    
    return g

#%% build fully connected graph
def build_fully_graph(N=30,mu=0,sigma=0.3):
    # v=mu+sigma*np.random.randn(N);
    # v=np.int_(np.floor(v)) # discrete attributes 
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
          # g.add_one_attribute(i,v[i])
          g.add_one_attribute(i,2)
          for j in range(N):
                if j != i:
                    g.add_edge((i,j))
                    
    return g

#%% build comunity graphs with different assortivity 
# pw is the possibility that one edge is connected 
def build_comunity_graph(N=30,mu=0,sigma=0.3,pw=0.8):
    v=mu+sigma*np.random.randn(N);
    v=np.int_(np.floor(v)) # discrete attributes 
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
         g.add_one_attribute(i,v[i])
         for j in range(N):
             if j != i:
                 r=np.random.rand()
                 if  r<pw:
                     g.add_edge((i,j))
                     
    return g

#%% merge community graphs
def merge_graph(g1,g2):  # inputs are nx_graph
    gprime=nx.Graph(g1)
    N0=len(gprime.nodes())
    g2relabel=nx.relabel_nodes(g2, lambda x: x +N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True)) 
    gprime.add_edge(N0-1,N0)

    return gprime

#%% build random graph G1
def build_G1(G,N=30,mu=0,sigma=0.3,pw=0.8):
    v=mu+sigma*np.random.randn(N);
    v=np.int_(np.floor(v)) # discrete attributes 
    
    L=len(G.nodes())
    G.add_nodes(list(range(N)))
    
    NN = N+L
    for i in range(L,NN):
          G.add_one_attribute(i,v[i-L])
    for i in range(NN):
          for j in range(NN):
                if j != i:
                    r=np.random.rand()
                    if  r<pw:
                      G.add_edge((i,j))
                      
    return G


#%%
        
for N_dum in NN_dum:
    Num = 1
    num = 0
    yes = 0
    DFGW= np.zeros(Num)
    while num<Num:
    
        plt.close("all")
        
        #%% 
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
        
        #%% merge 
        # n=0
        # Num=[]
        # while n<=3:
        #     num=np.random.randint(1,4) # randomly generate a number within [1,2,3]
        #     Num=np.append(Num,num)
        #     g1=merge_graph(g1, com_graph[num].nx_graph)
        #     n+=1
            
        # G1=Graph(g1)
        
        #%% build a fully connected graph (also the subgraph)
        # G11=build_fully_graph(N=N,mu=mu1,sigma=0.01)
        # G11 = build_star_graph()
        # G11=build_comunity_graph(N=N,mu=mu1,sigma=2, pw=0.5) 

        #%% build a random subgraoh
        G0 = Graph() # an empty graph
        np.random.seed(12)  # different graph with different seed -> same subgraph everytime
        G11 = build_G1(G0, N=N, mu=2, sigma = sigma1, pw = pw1) # set pw = 1 to build a fully-conn graph
        
        #%% build G1
        np.random.seed() # different graph G1 every time
        G12=copy.deepcopy(G11) #initialize with subgraph
        # G111=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G112=build_G1(G12,N=N2,mu=1,sigma=8,pw=0.1)
        # G1 = Graph(merge_graph(G111.nx_graph,G112.nx_graph))
        G1=build_G1(G12,N=N2,mu=2, sigma=sigma2, pw=pw2)
        
        # check if all nodes in G1 are connected
        # temp=G1.nx_graph._adj
        # if any(value=={} for value in temp.values()) == True:
        #     print("oops")
        #     continue
            
        #%%
        G2_nodummy=copy.deepcopy(G11)
        # G2_nodummy=build_fully_graph(N=25,mu=mu1,sigma=0.3)
        G2=copy.deepcopy(G2_nodummy)
        # G2.add_attributes({len(G2.nodes()): 0 })  # add dummy 
        i=0
        while i<N_dum:
            G2.add_attributes({len(G2.nodes()): 0 })
            i+=1
        
        #%%  The followings are fixed
        g1 = G1.nx_graph
        g2 = G2.nx_graph
        
        #%% check if every pair of nodes have path
        # n1 = len(G1.nodes())
        # try:
        #     for ii in range(n1):  
        #           nx.shortest_path_length(g1,source=0,target=ii)
        # except: 
        #     print("oops2")
        #     continue
                    
        #%%
        vmin=0
        vmax=9  # the range of color
        
        # plt.figure(figsize=(8,5))
        # draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
        # draw_rel(g2,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
        # plt.title('Two graphs. Color indicates the label')
        # plt.show()
        
        #%% weights and feature metric
        p1=ot.unif(len(G1.nodes()))
        p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2_nodummy.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
        # p2=np.append(p2_nodummy,1-sum(p2_nodummy))
        p2=np.append(p2_nodummy,
                     np.matlib.repmat(  (1-sum(p2_nodummy))/N_dum  , 1, N_dum)
                     )
        fea_metric = 'dirac'
        # fea_metric = 'hamming'
        # fea_metric = 'sqeuclidean'
        
        #%% use the function from FGWD all the time
        thresh=0.004
        # WD
        dw,log_WD,transp_WD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=0,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy,N_dum)
        # fig=plt.figure(figsize=(10,8))
        # plt.title('WD coupling')
        # draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.show()
        
        # GWD
        dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy,N_dum)
        # fig=plt.figure(figsize=(10,8))
        # plt.title('GWD coupling')
        # draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.show()
        
        # FGWD
        alpha=0.5
        dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2,p1,p2,p2_nodummy,N_dum)
        fig=plt.figure(figsize=(10,8))
        plt.title('FGWD coupling')
        draw_transp(G1,G2,transp_FGWD,N_dum,
                    shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        plt.show()
        
        #%% FGWD, find alpha
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
    
        #%%
        thre1=1e-5
        thre2=-0.015000 # entropic
    
        DFGW[num]=dfgw
        if dfgw<thre1:
            yes+=1
        num+=1
        print(num)
        
    print(yes/Num)
    print(np.std(DFGW))
    
    DFGW_set.append (DFGW)
    Percent.append (yes/Num)
    Mean.append (np.mean(DFGW))
    STD.append (np.std(DFGW))
    
#%%
fig, ax = plt.subplots()
ax.set_title('Hide Outlier Points')
ax.boxplot(DFGW_set, showfliers=False, showmeans=False)
#%% plot mean and STD
plt.figure()
plt.plot(np.array(NN_dum), np.array(Mean), 'k-')
plt.fill_between(np.array(NN_dum), np.array(Mean)-np.array(STD), np.array(Mean)+np.array(STD))
#%% plot percentage
plt.figure()
plt.plot(np.array(NN_dum),np.array(Percent))
plt.grid()
