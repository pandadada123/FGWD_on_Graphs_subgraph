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

vmin=-5
vmax=20 # the range of color

# for i in range(10):
#     x=X[i]
#     plt.figure(figsize=(8,5))
#     draw_rel(x.nx_graph,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
#     plt.title('Original graph Color indicates the label')
#     plt.show()

#%% create a query graph
G2=Graph()
# %% protein 
# G2.add_attributes({1:x.get_attr(1),
#                   12:x.get_attr(12),
#                   23:x.get_attr(23),
#                   33:x.get_attr(33),
#                   31:x.get_attr(31),
#                   41:x.get_attr(41)})  # without dummy

# G2.add_edge((1,12))
# G2.add_edge((1,33))
# G2.add_edge((1,23))
# G2.add_edge((12,23))
# G2.add_edge((12,33))
# G2.add_edge((23,31))
# G2.add_edge((23,41))
# G2.add_edge((33,41))
# G2.add_edge((33,31))

# G2.add_attributes({1:[23.0],
#                     2:[22.0],
#                     3:[3.0],
#                     4:[3.0],
#                     5:[3.0],
#                     6:[3.0]})  # without dummy

# G2.add_edge((1,2))
# G2.add_edge((1,3))
# G2.add_edge((1,4))
# G2.add_edge((2,3))
# G2.add_edge((2,4))
# G2.add_edge((3,5))
# G2.add_edge((3,6))
# G2.add_edge((4,5))
# G2.add_edge((4,6))

# G2.add_attributes({1:[8.0],
#                     2:[3.0],
#                     3:[5.0],
#                     4:[4.0],
#                     5:[7.0],
#                     6:[7.0]})  # without dummy
# G2.add_edge((1,2))
# G2.add_edge((1,3))
# G2.add_edge((2,3))
# G2.add_edge((2,4))
# G2.add_edge((3,4))
# G2.add_edge((3,5))
# G2.add_edge((4,5))
# G2.add_edge((4,6))
# G2.add_edge((5,6))

# x2 = X[16]
# G2.add_attributes({1:x2.get_attr(961),
#                     2:x2.get_attr(944),
#                     3:x2.get_attr(945),
#                     4:x2.get_attr(946),
#                     5:x2.get_attr(947),
#                     6:x2.get_attr(962)})  # without dummy
# G2.add_edge((1,2))
# G2.add_edge((1,3))
# G2.add_edge((2,3))
# G2.add_edge((2,4))
# G2.add_edge((3,4))
# G2.add_edge((3,5))
# G2.add_edge((4,5))
# G2.add_edge((4,6))
# G2.add_edge((5,6))

#%% bzr 1 
x2 = X[0]
G2.add_attributes({1:x2.get_attr(1),
                    2:x2.get_attr(2),
                    3:x2.get_attr(3),
                    4:x2.get_attr(4),
                    5:x2.get_attr(5),
                    6:x2.get_attr(6)})  # without dummy

G2.add_edge((1,2))
G2.add_edge((2,3))
G2.add_edge((3,4))
G2.add_edge((4,5))
G2.add_edge((5,6))
G2.add_edge((6,1))

# G2.add_edge((1,3))
# G2.add_edge((1,5))
# G2.add_edge((2,4))
# G2.add_edge((2,5))
# G2.add_edge((2,6))
# G2.add_edge((3,5))
# G2.add_edge((3,6))
# G2.add_edge((4,6))

# G2.add_attributes({1:x.get_attr(1),
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

# G2.add_edge((1,2))
# G2.add_edge((2,3))
# G2.add_edge((3,4))
# G2.add_edge((4,5))
# G2.add_edge((5,6))
# G2.add_edge((6,1))

# G2.add_edge((4,7))
# G2.add_edge((7,8))
# G2.add_edge((8,9))
# G2.add_edge((9,10))
# G2.add_edge((10,11))
# G2.add_edge((11,5))
#%%
g2=G2.nx_graph

plt.figure(figsize=(8,5))
draw_rel(g2,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
plt.axis('off')
plt.show()
    
    #%%
obj=[]
# for i in range(663,1113):
# for i in range(len(X)):
for i in range(1):
    # if label[i]==1:
    #     dfgw = np.nan
    #     print(i)
        
    # elif label[i]==-1:
        x=X[36]
       
        # x=X[16]
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
        def change(graph):  # change attribute of list to array AND change the node indexes to start from 0
        
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
        
       
        # G2 = Graph(change(g2))
        # g2=G2.nx_graph
        #%%
        G1=Graph(g1)
        G2_dummy=copy.deepcopy(G2)
        # G2_dummy.add_attributes({100: np.array ( [100]*(len(x.get_attr(1)))  )  })  # add dummy 
        G2_dummy.add_attributes({1e6: np.array([0])  })  # add dummy 
        # G2_dummy.add_attributes({len(G2.nodes()): [0] })  # add dummy 
        g2_dummy=G2_dummy.nx_graph
        
        # G2_dummy = Graph(change(g2_dummy))
        
        #%%
        # g1=G1.nx_graph
        # g2=G2.nx_graph
        # g2_dummy=G2_dummy.nx_graph
            
        
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
        p2_nodummy=1/len(G1.nodes()) * np.ones([len(G2.nodes())])    # ACTUALLY NOT USED IN THE ALGORITHM
        p2=np.append(p2_nodummy,1-sum(p2_nodummy))
        
        # dw,transp_WD=Wasserstein_distance(features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
        # # dw=Wasserstein_distance(features_metric='dirac').graph_d(g1,g2)
        thresh=0.002
        # plt.title('WD coupling')
        # draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        # plt.show()
        
        ## dgw=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='dirac',method='shortest_path').graph_d(g1,g2)
        # dgw,log_GWD,transp_GWD=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='sqeuclidean',method='shortest_path',loss_fun= 'square_loss').graph_d(G1,G2_dummy,p1,p2,p2_nodummy)
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
        
        dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric= fea_metric, method='shortest_path', loss_fun='square_loss').graph_d(G1,G2_dummy,p1,p2,p2_nodummy)
        plt.figure(figsize=(8,5))
        # plt.title('FGWD coupling')
        draw_transp(G1,G2_dummy,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
        plt.axis('off')
        plt.show()
    
        # plt.close("all")
        
        #%%
        if len(x.nodes())<len(G2.nodes()):
            dfgw = np.nan
            
        obj.append(dfgw)
        
        print (i)
    
#%%
print(obj)

Thre1=1e-5
index1 = []
obj1 = []
for idx in range(len(obj)) :
    if obj[idx] < Thre1:
        index1.append(idx)
        obj1.append(obj[idx])
print(index1)
print(obj1)

Thre2=1e-3
index2 = []
obj2 = []
for idx in range(len(obj)) :
    if obj[idx] < Thre2:
        index2.append(idx)
        obj2.append(obj[idx])
print(index2)
print(obj2)
