# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:24:58 2023

@author: Pandadada
"""

# Finding subgraph

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/OT_sim/FGW-master/FGW-master/lib'))

# from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from graph import Graph
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
import copy
from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib import cm
import networkx as nx
import ot

#%% Define two graphs 
G1=Graph()
G1.add_attributes({0:1,1:7,2:5})    # add color to nodes
G1.add_edge((0,1))
G1.add_edge((1,2))
G1.add_edge((2,0))

G2=Graph()
G2.add_attributes({0:1,1:7})
G2.add_edge((0,1))

g1=G1.nx_graph
g2=G2.nx_graph

#%%
def graph_colors(nx_graph,vmin=0,vmax=7):
    cnorm = mcol.Normalize(vmin=vmin,vmax=vmax)
    cpick = cm.ScalarMappable(norm=cnorm,cmap='viridis')
    cpick.set_array([])
    val_map = {}
    for k,v in nx.get_node_attributes(nx_graph,'attr_name').items():
        val_map[k]=cpick.to_rgba(v)  # ADD IF HERE
        # val_map[k]=cpick.to_rgba(sum(pow(np.array(v),2)))
    colors=[]
    for node in nx_graph.nodes():
        colors.append(val_map[node])
    return colors

def draw_rel(G,draw=True,shiftx=0,shifty=0,return_pos=False,with_labels=True,swipy=False,swipx=False,vmin=0,vmax=7):

    pos=nx.kamada_kawai_layout(G) # the layout is almost optimal
    
    if shiftx!=0 or shifty!=0:
        for k,v in pos.items():
            # Shift the x values of every node by shiftx,shifty
            if shiftx!=0:
                v[0] = v[0] +shiftx
            if shifty!=0:
                v[1] = v[1] +shifty
            if swipy:
                v[1]=-v[1]
            if swipx:
                v[0]=-v[0]

    colors=graph_colors(G,vmin=vmin,vmax=vmax)  # return RGBA colors of the nodes, vmin and vmax are the range indexes
    if with_labels:
        # nx.draw(G,pos,with_labels=True,labels=nx.get_node_attributes(G,'attr_name'),node_color = colors)  # put color values on nodes
        nx.draw_networkx(G,pos,node_color = colors)  # put keys/indexes on nodes
    else:
        nx.draw(G,pos,with_labels=False,node_color = colors)
    if draw:
        plt.show()
    if return_pos:
        return pos
        
#%%
vmin=0
vmax=9  # the range of color
plt.figure(figsize=(8,5))
draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
draw_rel(g2,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
plt.title('Two graphs. Color indicates the label')
plt.show()

#%%
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d

# from FGW import tensor_matrix,gwloss

A = np.linspace(0,1/3,50)
B = np.linspace(0,1/3,50)

y = np.zeros([len(A),len(B)])
yy = np.zeros([len(A),len(B)])

C1=np.array([[0,1,1],[1,0,1],[1,1,0]])
C2=np.array([[0,1],[1,0]])

L=np.zeros([len(C1),len(C1[1]),len(C2),len(C2[1])]) # L is a 4-dim tensor constant            
for i in range(len(C1)):
    for ii in range(len(C1[1])):         
            for j in range(len(C2)):
                for jj in range(len(C2[1])):   # Only do alignment here, no element is zero
                               c1=C1[i][ii]
                               c2=C2[j][jj]
                               # np.append(L,pow((c1-c2),2))    
                               L[i][ii][j][jj]=pow((c1-c2),2) 
                               
                               
def tensor_matrix(L,T):
    S=np.shape(L)
    opt_tensor=np.zeros([S[0],S[2]])
    for i in range(S[0]):
        for ii in range(S[1]):         
                for j in range(S[2]):
                    for jj in range(S[3]):
                        opt_tensor[i][j]+=L[i][ii][j][jj]*T[ii][jj]
    return opt_tensor

def gwloss(L,T):
    return np.sum( tensor_matrix(L, T) * T)
                               
for i in range(len(A)):
    for j in range(len(B)):
        a=A[i]
        b=B[j]
        T=np.array([[a,1/3-a],[b,1/3-b],[1/2-a-b,-1/6+a+b]])
        y[i][j]=gwloss(L,T)
        yy[i][j] = 2/3+1/2-2*( (1/2-a)*(1/3-a)+(1/6+a)*a+(1/2-b)*(1/3-b)+(1/6+b)*b+(a+b)*(-1/6+a+b)+(2/3-a-b)*(1/2-a-b) ) # formula by hand
        
        
#%%
y[np.isnan(y)] = 0
#%% plot surface
fig = plt.figure()
ax = plt.axes(projection='3d')
AA, BB = np.meshgrid(A, B)
surf = ax.plot_surface(AA,BB,y, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()

#%% contour 等高线
fig = plt.figure()
levels = np.arange (-3/18, np.max(y), 1/18)
# h = plt.contourf(A, B, y, levels = levels , cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
h = plt.contour(A, B, y, levels=levels, cmap=cm.coolwarm)
plt.clabel(h, inline=1, fontsize=10, colors='k')
plt.axis('scaled')
# plt.colorbar()

# plot feasible set
B1 = 1/6-A
B2 = 1/2-A
plt.plot(A,B1,A,B2, color = 'b', linewidth=2, linestyle="--")

def plot_seg(point1,point2):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color = 'b', linewidth=2, linestyle="--")
    
plot_seg([0,1/6],[0,1/3])
plot_seg([0,1/3],[1/6,1/3])
plot_seg([1/6,0],[1/3,0])
plot_seg([1/3,0],[1/3,1/6])

# six notated points

# plt.plot([0], [1/6] , 'o')
plt.annotate('A', xy=(0,1/6), xytext=(-0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('B', xy=(0,1/3), xytext=(-0.02, 1/3), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('C', xy=(1/6,1/3), xytext=(1/6, 1/3+0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('D', xy=(1/3,1/6), xytext=(1/3+0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('E', xy=(1/3,0), xytext=(1/3+0.02, 0), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('F', xy=(1/6,0), xytext=(1/6,-0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))

# plt.annotate('A', xy=(0,1/6), xytext=(-0.03, 1/6), color = 'b')

# x = [0, 0, 1/6, 1/3, 1/3, 1/6]
# y = [1/6, 1/3, 1/3, 1/6, 0, 0]
# annotations = ["A", "B", "C", "D", "E", "F"]
# plt.scatter(x, y, s=100)

# for xi, yi, text in zip(x, y, annotations):
    # plt.annotate(text,
    #             xy=(xi, yi), xycoords='data',
    #             xytext=(1.5, 1.5), textcoords='offset points')
    # plt.annotate(text, xy=(xi, yi), xytext=(-0.03, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.03))
    
    
plt.show()






#%% compare GWD and FGWD

p1=ot.unif(3)
p2=ot.unif(2)

def draw_transp(G1,G2,transp,shiftx=1,shifty=0,thresh=0.09,swipy=False,swipx=False,vmin=0,vmax=7,with_labels=True):
    pos1=draw_rel(G1.nx_graph,draw=False,return_pos=True,vmin=vmin,vmax=vmax,with_labels=with_labels)
    pos2=draw_rel(G2.nx_graph,draw=False,shiftx=shiftx,shifty=shifty,return_pos=True,swipx=swipx,swipy=swipy,vmin=vmin,vmax=vmax,with_labels=with_labels)
    _,invd1=G1.all_matrix_attr(return_invd=True)
    _,invd2=G2.all_matrix_attr(return_invd=True)
    for k1,v1 in pos1.items():
        for k2,v2 in pos2.items():
            if (transp[invd1[k1],invd2[k2]]>thresh):
                plt.plot([pos1[k1][0], pos2[k2][0]]
                          , [pos1[k1][1], pos2[k2][1]], 'r--'
                          , alpha=transp[invd1[k1],invd2[k2]]/np.max(transp),lw=2)

thresh=0.004
                
# dw,transp_WD=Wasserstein_distance(features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
# # dw=Wasserstein_distance(features_metric='dirac').graph_d(g1,g2)
# plt.title('WD coupling')
# draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()

fig = plt.figure()
# dgw=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='dirac',method='shortest_path').graph_d(g1,g2)
dgw,transp_GWD=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='sqeuclidean',method='shortest_path').graph_d(G1,G2,p1,p2)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()


alpha=0.5
fig = plt.figure()
# dfgw,transp_FGWD=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric='dirac',method='shortest_path').graph_d(g1,g2,p1,p2)
dfgw,transp_FGWD=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric='sqeuclidean',method='shortest_path').graph_d(G1,G2,p1,p2)
plt.title('FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

