# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:59:13 2023

@author: Pandadada
"""


# coding: utf-8

# # Tutorial

# In[1]:
# import numpy as np
# import os
# import sys
# sys.path.append(os.path.realpath(
#         'E:/Master Thesis/comparisons/fornax2/fornax'))

import sys
sys.modules[__name__].__dict__.clear()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
import fornax

import random
import string
import pickle
import time

from lib1.graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
from lib1.data_loader import load_local_data,histog,build_noisy_circular_graph
import copy

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import SVG

# Add project root dir
# ROOT_DIR = os.path.abspath("../../")
# sys.path.append(ROOT_DIR)

from sqlalchemy import text


# To install the use the dependencies for this notebook:
# 
# ```bash
# conda env create -f environment.yml
# source activate fornax_tutorial
# ```
# 
# To run this notebook from the project root:
# 
# ```bash
# cd docs/tutorial
# jupyter-notebook
# ```

# In this tutorial we will:
# 
# * Load a graph of superheros and their teams from csv files
# 
# * Search for nodes in the graph using a string similarity function
# 
# * Use fornax to search for nodes using string similarity and fuzzy graph matching
# 
# The data in this tutorial we be generated using the preceding notebook: `Tutorial1.ipynb`.
# 
# ## Introduction
# 
# `nodes.csv` and `edges.csv` contain a graph of superheros and their teams along with alternative names for those heros and groups (or aliases).
# 
# The image below uses the example of Iron Man, who is known as "Tony" to his friends.
# Iron man is a member of the Avengers, a.k.a. Earth's Mightiest Superheros.
# Other heros are also members of The Avengers, and they will also have aliases.
# Other heros will also be members of other teams and so and so forth.
# 
# 
# All of these heros, teams and aliases together make our target graph, a graph which we will search using 

# In[2]:


# SVG('../img/iron_man.svg')
# SVG('E:/Master Thesis/comparisons/iron_man.svg')


# Let's load the data into the notebook using pandas.

# In[3]:

#%%

# # used for converting csv values in nodes.csv
# mapping = {
#     '0': 'hero',
#     '1': 'team', 
#     '2': 'hero_alias', 
#     '3': 'team_alias'
# }


# nodes_df = pd.read_csv(
#     './nodes.csv', 
#     # rename the columns as targets as this will form the target graph
#     # (the graph which we will be searching)
#     names=['target_label', 'target_type', 'target_id'],
#     # ignore the header
#     header=0,
#     converters = {
#         # convert target_type from numeric values to
#         # literal string representations for ease of reading
#         'target_type': lambda key: mapping.get(key)
#     }
# )

# # contains pairs of target node ids
# edges_df = pd.read_csv('./edges.csv')


# # We can see that the target nodes have a label (the hero's primary name).
# # The target_type column will be one of `hero`, `team`, `hero alias`, `team alias`, the four types of nodes in the graph.
# # 
# # (Note that by hero we mean a person in a comic book who has superpowers regardless of them being good or bad)

# # In[4]:


# nodes_df['target_label'].head()


# # Edges are pairs of `target_id` values.
# # Note that fornax deals with undirected graphs so there is no need to add the edge in the reverse direction.
# # Doing so will cause an exception as the edge will be considered a duplicate.

# # In[5]:


# edges_df.head()

#%%

# ## Label similarity
# 
# For some motivation, before using fornax, let us search for nodes just using their labels.
# Let's search for nodes similar to `guardians`, `star` and `groot`.
# 
# We will create a function that given a pair of labels, it will return a score where:
# 
# $$0 <= score <= 1$$
# 
# Secondly we'll create a search function that returns rows from our table of target nodes that have a non zero similarity score.

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
# NN3 = [15]
# NN3 = [20,50,100,200,300,400,500]
NN3 = [50, 100, 1000, 3000, 5000, 7000, 10000]
# NN3 = [15,20,25]
# NN3 = [15,45,75]
# N3 = N+N2
# N3 = 45
# NN3 = [45]
# Pw = np.linspace(0.1, 1, 10)
# Pw = np.linspace(0.01, 0.1, 10)
d = 2
# Deg = [0.5,1]+[x for x in range(2, 15, 2)]
# Deg = [10]
# Deg = [0.5,1]
deg = 3
# deg = 0.1
# Pw2 = [deg / (N3-1) for deg in Deg]
# Pw = [0.1]
# pw1 = 0.5 # query
# pw1 = d / (N-1)
# pw1 = np.random.choice(np.linspace(0.1, 1, 10))
# pw2 = 0.5 # target
# pw1 = deg / (N-1)
# pw2 = deg / (N3-1)
# Sigma2=[0.01,0.1,0.5,1,2,3,4]
# Sigma2=[0.01]
# sigma1=0.1
# sigma2=0.1
numfea = 20
# NumFea = list(range(1, 20))  # from 1 to 20
# NumFea = [x for x in range(2, 41, 2)]
# NumFea = [2]

# Alpha = np.linspace(0, 1, 11)

# Dia = [i for i in range(1, N)]

thre1 = 1e-9
# thre2=-0.015000 # entropic
thre2 = 1e-2
thre3 = 0.05
epsilon = thre1
        
Is_fig = 0
Is_info = 0
Is_fea_noise = 0
Is_str_noise = 0

Num = 10 # number of repeats (generate a random graph and a query)
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

#%% add noise to the query
def add_noise_to_query(g,fea_metric,
                       mean_fea,std_fea,str_mean,str_std,
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

        elif fea_metric == 'dirac' or fea_metric == 'sqeuclidean':
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


Is_create_query = 0

mean_fea = 1 # number of nodes that has been changed
std_fea = 0.5 # zero mean Gaussian
str_mean = 0
str_std = 0.1
# Generate a random string of given length
def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# # Create a list of 97 random strings
# target_labels = query_labels + [random_string(random.randint(1, 10)) for _ in range(97)]
# target_labels = query_labels + [random.choice(query_labels) for _ in range(100-5)]
# target_labels = ['guardians', 'groot', 'star', 'guardians2']

Eps = 1e-9

# missing_files_count = 0
# Cost = np.zeros(NumQ)
# Ratio = np.zeros(NumQ)
# Time_execute = np.zeros(NumQ)
# Time_match = np.zeros(NumQ)
# Time_opt = np.zeros(NumQ)
# Time_total = np.zeros(NumQ)
# Time_total_2 = np.zeros(NumQ)

Ratio_set = []
Time_total_2_set = []
Index_set = []
Rate_exact_set = []
Cost_set = []
Rate1_subopt_set = []
Rate3_subopt_set = []

for N3 in NN3:
    
    num = 0
    
    missing_files_count = 0
    Cost = np.zeros(Num)
    Ratio = np.zeros(Num)
    Time_execute = np.zeros(Num)
    Time_match = np.zeros(Num)
    Time_opt = np.zeros(Num)
    Time_total = np.zeros(Num)
    Time_total_2 = np.zeros(Num)
    Index = np.zeros(Num)
    
    while num < Num: 
        
        print("num=",num)
        
        # %% build G1
        # pw1=0.1
        # G11 = build_comunity_graph(N=N, numfea=numfea, pw=pw1, fea_metric=fea_metric)
        
        # build line graph for query
        G11 = build_line_graph(N=N, numfea=numfea, fea_metric=fea_metric)
        
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
        
        #%% add noise to query
        if Is_fea_noise or Is_str_noise:
            g2_nodummy = add_noise_to_query(g2_nodummy, fea_metric=fea_metric, mean_fea = mean_fea, std_fea = std_fea, str_mean= str_mean, str_std= str_std,
                                   Is_fea_noise=Is_fea_noise, Is_str_noise=Is_str_noise)            
        
        G2_nodummy = Graph(g2_nodummy)
        
        #%% only allow the query is connected
        is_connected = nx.is_connected(g2_nodummy)
        if is_connected == 0:
            print("'The query graph is not connected.'")
            continue
        
        #%%
        nodes1=g1.nodes() # [dict]
        nodes2=g2_nodummy.nodes()
        
        target_id = sorted(list(g1.nodes.keys()))  # [list] order of nodes for cost matrices, from small to large
        query_id = sorted(list(g2_nodummy.nodes.keys()))
        
        target_labels = []
        query_labels_original = []
        for i in range(N3):                
            key1 = target_id[i]
            f1 = nodes1[key1]['attr_name']
            target_labels.append(f1)
            
        for i in range(N):                
            key2 = query_id[i]
            f2 = nodes2[key2]['attr_name']
            query_labels_original.append(f2)
            
        target_edges = list(g1.edges())
        query_edges_original = list(g2_nodummy.edges())
        
        #%% noisy
        query_edges = query_edges_original
        query_labels = query_labels_original
            
        #%% change the ids of this graph
        def relabel_graph(ids, edges):
            # Create a mapping from old node IDs to new node IDs
            old_to_new = {node: idx for idx, node in enumerate(ids)}
        
            # Create a list of new edges with updated node IDs
            new_edges = [(old_to_new[start], old_to_new[end]) for start, end in edges]
        
            # Create a list of new ids, simply using the range of new IDs
            new_ids = list(range(len(ids)))
        
            return new_ids, new_edges
        
        target_id, target_edges = relabel_graph(target_id, target_edges)
        query_id, query_edges = relabel_graph(query_id, query_edges)
        
        #%%
        # query_labels = ['guardians', 'groot', 'star','a','a'] # 4 features, 5 nodes
        
        # create nodes_df by hand 
        # Create a dictionary with the desired data
        data1 = {
            'Index': list(range(len(target_labels))),
            'target_label': target_labels,
            'target_type': ['hero'] * len(target_labels),
            'target_id': target_id
            # 'target_id': list(x+1000 for x in list(range(len(target_labels))))
            # 'target_id': list(range(len(target_labels)))
            # 'target_id': [1081675, 1276753309, 74671434, 1295400389]
        
        }
        
        # Create the DataFrame
        nodes_df = pd.DataFrame(data1)
        
        #%%
        # start_range = 1000
        # end_range = 1100
        
        # # Create a list of integers in the specified range
        # integers = list(range(start_range, end_range + 1))
        
        # # Shuffle the integers to ensure randomness
        # # random.shuffle(integers)
        
        # # Create two lists of length 
        # # list_a = [1000,1000]
        # # list_b = [1001,1002]
        # list_a = []
        # list_b = []
        
        # for a in range(1000,1100):
        #     # a = random.choice(integers)
        #     # b = random.choice(integers)
        #     # a = integers[_+1]
        #     for b in range(a+1,1100):
        #         list_a.append(a)
        #         list_b.append(b)
        
        # Data for the DataFrame (the ids should be the same as data1)
        # data2 = {
        #     'Index': list(range(len(list_a))),  # First column with indexes starting from 0
        #     # 'start': [100, 100, 100,100],  # Second column with the specified values
        #     # 'end': [101, 102, 103,101]     # Third column with the specified values  
        #     'start': list_a,
        #     'end': list_b
        #     # 'start': [0, 0, 0],  # Second column with the specified values
        #     # 'end': [1, 2, 3]     # Third column with the specified values  
        #     # 'start': [1081675, 1081675, 1081675],  # Second column with the specified values
        #     # 'end': [1276753309, 74671434, 1295400389]     # Third column with the specified values 
        # }
        
        
        start_ids = [edge[0] for edge in target_edges]
        end_ids = [edge[1] for edge in target_edges]
        
        
        data2 = {
            'Index': list(range(len(start_ids))),  # First column with indexes starting from 0 # edges indices
            # 'start': [100, 100, 100,100],  # Second column with the specified values
            # 'end': [101, 102, 103,101]     # Third column with the specified values  
            'start': start_ids, # start node index
            'end': end_ids # end node index 
        }
        # Create the DataFrame
        edges_df = pd.DataFrame(data2)
        
        #%%
        # def node_scoring_function(first: str, second: str):
        #     """ node scoring function takes two strings and returns a 
        #         score in the range 0 <= score <= 1
        #     """
        #     first_, second_ = sorted((first.lower(), second.lower()), key=len)
        #     # if first is not a substring of second: score = 0
        #     if not first_ in second_:
        #         return 0
        #     # otherwise use the relative difference between
        #     # the two lengths
        #     score = len(second_) - len(first_)
        #     score /= max(len(first_), len(second_))
        #     score = 1. - score
        #     return score
        
        # def node_scoring_function(f1, f2, fea_metric): # return 0 when two features are the same
        #     if fea_metric == 'jaccard':
        #         set1 = set(f1)
        #         set2 = set(f2)
        #         intersection = len(set1.intersection(set2))
        #         union = len(set1.union(set2))
        #         return 1.0 - intersection / union
            
        #     elif fea_metric == 'sqeuclidean':        
        #         if f1.shape != f2.shape:
        #             return 1.0
        #         else: 
        #             return 1-1/(1+np.sum([pow(f1-f2 ,2)]))
        
        # def node_scoring_function(f1, f2): # return 0 when two features are the same
        #      # fea_metric == 'sqeuclidean':        
        #     f1 = np.array(f1)
        #     f2 = np.array(f2)
        #     if f1.shape != f2.shape:
        #         return 1.0
        #     else: 
        #         return 1-1/(1+np.sum([pow(f1-f2 ,2)]))
        
        def node_scoring_function(f1, f2, fea_metric): # return 1 when two features are the same
            if fea_metric == 'sqeuclidean':        
                f1 = np.array(f1)
                f2 = np.array(f2)
                if f1.shape != f2.shape:
                    return 0
                else: 
                    return 1/(1+np.sum([pow(f1-f2 ,2)]))
            
            elif fea_metric == 'dirac':
                if f1 == f2:
                    return 1
                else:
                    return 0
                    
        # In[7]:
        # def search(query_id: int, query_label: str):
        def search(query_id, query_label, Eps):
            # compute all of the scores
            scores = nodes_df['target_label'].apply(
                node_scoring_function, 
                args=(query_label,fea_metric)
            )
            
            # to include all labels in the candidate list
            small_value = 1e-20
            scores = scores.replace(0, small_value)
            
            # create a boolean mask
            # epsilon = 1e-1 # for noisy query
            # epsilon = 1e-9 # for clean query
            epsilon = Eps
            mask = scores > 1-epsilon
            # mask = scores > 0
            
            # # If we do not want to set constraints on features
            # mask = scores >= 0
            
            # graph the non zero scoring nodes
            matches = nodes_df[mask].copy()
            # add extra columns
            matches['score'] = scores[mask]
            # matches['query_label'] = query_label
            matches['query_label'] = [query_label]*len(matches)
            matches['query_id'] = query_id
            return matches
        
        
        # ### Aside:
        # Note that these string search functions are not terribly efficient.
        # They involve repeated full scans of the target nodes table.
        # If we were searching a larger graph we could use a search tree as an index, an external sting matching service or database. However, since this is a tutorial, the above functions are simpler and more reproducible.
        # This is important as we will be using these search results with 
        
        # In[8]:
        
        
        # Examining the table below we can see that we have a conundrum.
        # There are 22 nodes with varying similarity to `star` and 4 nodes similar to `galaxy`.
        
        # In[9]:
        
        
        # find the nodes similar to 'guardians', 'star' and 'groot'
        start_match = time.time()
        matches = pd.concat(search(id_, label, Eps) for id_, label in enumerate(query_labels))
        end_match = time.time()
        matches
        Time_match[num]=end_match - start_match
        print("TimeMatch",Time_match[num])
        
        # unique_count = matches['query_id'].nunique() # no need to add this
        # if unique_count < N:
        #     print("no enough nodes in the candidate set")
        #     Ratio[num]=np.nan
        #     Time[num]=np.nan
        #     continue
    
        
        # Fornax enables a more powerful type of search. 
        # By specifying 'guardians', 'star', 'groot' as nodes in a graph, 
        # and by specifying the relationships between them, 
        # we can search for nodes in our target graph with the same relationships.
        
        # ## Creating a target graph
        # 
        # Fornax behaves much like a database. In fact it uses SQLite or Postgresql to store graph data and index it.
        # To insert a new graph into fornax we can use the following three steps:
        # 1. create a new graph
        # 2. add nodes and node meta data
        # 3. add edges and edge meta data
        # 
        # The object `GraphHandle` is much like a file handle. It does not represent the graph but it is an accessor to it.
        # If the `GraphHandle` goes out of scope the graph will still persist until it is explicitly deleted, much like a file.
        
        # In[10]:
        
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:
            target_graph = fornax.GraphHandle.create(conn)
            target_graph.add_nodes(
                # use id_src to set a custom id on each node 
                id_src=nodes_df['target_id'],
                # use other keyword arguments to attach arbitrary metadata to each node
                label=nodes_df['target_label'],
                # the type keyword is reserved to we use target_type
                target_type=nodes_df['target_type']
                # meta data must be json serialisable
            )
            target_graph.add_edges(edges_df['start'], edges_df['end'])
            
            # target_graph.graph_id = None
            
        # We can use the `graph_id` to access our graph in the future.
        
        # In[11]:
        
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:
            target_graph.graph_id
            another_target_graph_handle = fornax.GraphHandle.read(conn, target_graph.graph_id)
            print(another_target_graph_handle == target_graph)
        
        
        # ## Creating a query graph
        # 
        # Let's imagine that we suspect `groot` is directly related to `guardians` and `star` is also directly related to `guardians`.
        # For example `groot` and `star` could both be members of a team called `guardians`.
        # Let's create another small graph that represents this situation:
        
        # In[12]:
        
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:
            # create a new graph
            query_graph = fornax.GraphHandle.create(conn)
        
            # insert the three nodes: 
            #   'guardians' (id=0), 'star' (id=1), 'groot' (id=2)
            query_graph.add_nodes(label=query_labels)
        
            # alternatively:
            #    query_graph.add_nodes(id_src=query_labels)
            # since id_src can use any unique hashable items
        
            # edges = [
            #     (0, 1), # edge between groot and guardians
            #     (0, 2)  # edge between star and guardians
            # ]
            
            # if target graph is fully connected, so as the query graph
            # edges = [ (0, 1), (0,2) , (0,3), (0,4),
            #          (1,2), (1,3), (1,4),
            #          (2,3), (2,4),
            #          (3,4)
            # ]
            
            edges = query_edges
            
            # # Create a mapping from old query_id values to new ones
            # query_id_mapping = {old_id: new_id for new_id, old_id in enumerate(set(query_id))}
            # # Update query_id using the mapping
            # query_id = [query_id_mapping[old_id] for old_id in query_id]
            # # Update edges using the updated query_id values
            # edges = [(query_id_mapping[source], query_id_mapping[target]) for source, target in edges]
            
            sources, targets = zip(*edges)
            
            query_graph.add_edges(sources, targets)
        
            # query_graph.graph_id = None
        
        # ## Search
        # 
        # We can create a query in an analogous way to creating graphs using a `QueryHandle`,
        # a handle to a query stored in the fornax database.
        # To create a useful query we need to insert the string similarity scores we computed in part 1.
        # Fornax will use these scores and the graph edges to execute the query.
        
        # In[13]:
        
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:
            query = fornax.QueryHandle.create(conn, query_graph, target_graph)
            query.add_matches(matches['query_id'], matches['target_id'], matches['score'])
            
            # query.qeury_id = None
        
        
        # Finally we can execute the query using a variety of options.
        # We specify we want the top 5 best matches between the query graph and the target graph.
        
        # In[14]:
        
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:
            
            start_time = time.time()
            results = query.execute(n=1, hopping_distance=1)  # top-n results
            end_time = time.time()
            Time_execute[num] = end_time - start_time
            
        # ## Visualise
        # 
        # `query.execute` returns an object describing the search result.
        # Of primary interest is the `graph` field which contains a list of graphs in `node_link_graph` format.
        # We can use networkx to draw these graphs and visualise the results.
        
        # In[15]:
        
        
        def draw(graph):
            """ function for drawing a graph using matplotlib and networkx"""
            
            # each graph is already in node_link_graph format 
            G = nx.json_graph.node_link_graph(graph)
            
            labels = {node['id']: node['label'] for node in graph['nodes']}
            node_colour = ['r' if node['type'] == 'query' else 'b' for node in graph['nodes']]
            pos = nx.spring_layout(G)
            
            nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colour, alpha=.3)
            
            edgelist = [(e['source'], e['target']) for e in graph['links'] if e['type'] != 'match']
            nx.draw_networkx_edges(G, pos, width=3, edgelist=edgelist, edge_color='grey', alpha=.3)
            
            edgelist = [(e['source'], e['target']) for e in graph['links'] if e['type'] == 'match']
            nx.draw_networkx_edges(G, pos, width=3, edgelist=edgelist, style='dashed', edge_color='pink')
            
            nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', labels=labels)
        
        
        # Result 1 contains the best match. The three query nodes (in red) best match the three target nodes (in blue). The dashed lines show which pairs of query and target nodes matched each other. The blue nodes are a subgraph of the target graph. Note that the result does not describe the whole target graph because in principle it can be very large.
        # 
        # Here we can see that the blue subgraph has exactly the same shape as the red query graph. However, the labels are not exactly the same (e.g. `guardians != Guardians of the Galaxy`) so the result scores less than the maximum score of 1.
        # However, we can see that our query graph is really similar to Groot and Star-Lord from Guardians of the Galaxy.
        # Since this is the best match we know that 
        
        # # In[16]:
        
        
        # for i, graph in enumerate(results['graphs'][:1]):
        #     # plt.title('Result {0}, score: {1:.2f}'.format(1, 1. - graph['cost']))
        #     plt.title('Result {0}, score: {1:.2f}'.format(1, graph['cost']))
        #     draw(graph)
        #     plt.xlim(-1.2,1.2)
        #     plt.ylim(-1.2,1.2)
        #     plt.axis('off')
        #     plt.show()
        
        
        # # Results 2-4 have a lower score because `star` matches to a different node not adjacent to Guardians of the Galaxy. Further inspection would show that `star` has matched aliases of Star-Lord which are near Guardians of the Galaxy but not ajacent to it.
        
        # # In[17]:
        
        
        # for i, graph in enumerate(results['graphs'][1:4]):
        #     # plt.title('Result {0}, score: {1:.2f}'.format(i+2, 1. - graph['cost']))
        #     plt.title('Result {0}, score: {1:.2f}'.format(i+2, graph['cost']))
        #     draw(graph)
        #     plt.xlim(-1.2,1.2)
        #     plt.ylim(-1.2,1.2)
        #     plt.axis('off')
        #     plt.show()
        
        
        # # The final match pairs `guardians` and `star` to two nodes that do not have similar edges to the target graph. `groot` is not found in the target graph. The result gets a much lower score than the preceding results and we can be sure that any additional results will also be poor because the result are ordered.
        
        # # In[18]:
        
        
        # for i, graph in enumerate(results['graphs'][4:]):
        #     # plt.title('Result {0}, score: {1:.2f}'.format(i+5, 1. - graph['cost']))
        #     plt.title('Result {0}, score: {1:.2f}'.format(i+5, graph['cost']))
        #     draw(graph)
        #     plt.xlim(-1.2,1.2)
        #     plt.ylim(-1.2,1.2)
        #     plt.axis('off')
        #     plt.show()
        
        #%%
        for i, graph in enumerate(results['graphs'][:1]):        
            edgelist = [(e['source'], e['target']) for e in graph['links'] if e['type'] == 'match']
            labels = {node['id']: node['label'] for node in graph['nodes']} # all labels include target and query
            Cost[num] = graph['cost']
            # ratio = 1
            query_labels_original 
            query_labels
            
            #%%
            # Step 1: Create lists list_A and list_B
            # list_A = [labels[node_id] for node_id, _ in edgelist]
            # list_B = [labels[node_id] for _, node_id in edgelist] # the latter element is the target (big graph) id (list_B is the same as labels)
            # list_BB = [node_id for _, node_id in edgelist] # id of the target 
            
            edgelist_B = [(e['source'], e['target']) for e in graph['links'] if e['type'] == 'target']
            labels_B = {node['id']: node['label'] for node in graph['nodes'] if node['type'] == 'target'} # dict of target id and target features 

            # Convert data to graphs
            g3 = nx.Graph()
            g3.add_edges_from(edgelist_B)
            nx.set_node_attributes(g3, labels_B, 'attr_name')
            
            # Check if the two graphs are isomorphic considering node labels
            Is_isomorphic = nx.is_isomorphic(g3, g2_nodummy, node_match=lambda n1, n2: n1['attr_name'] == n2['attr_name'])
            
            print(Is_isomorphic)  # True or False

            # # Step 2: Find indices of elements in list_A that are in query_labels
            # indices = [i for i, label in enumerate(list_A) if label in query_labels]
            
            # # Step 3: Reorder query_labels_original based on indices
            # query_labels_original_reorder = [query_labels_original[i] for i in indices]
            
            # # Step 4: Compare list_B and query_labels_original_reorder
            # matching_count = sum(1 for x, y in zip(list_B, query_labels_original_reorder) if x == y)
            
            # # Print the count of matching elements
            # print("Number of matching elements between list_B and query_labels_original_reorder:", matching_count)
            #%% Ratio of nodes that are matched
            # ratio = matching_count / len(edgelist)
            
        #%%
        Index[num] = Is_isomorphic
        # Ratio[num] = ratio
        
        Time_opt[num] = results['time']
        
        # print('ratio', Ratio[num])
        print('time_execute', Time_execute[num])
        print('time_match', Time_match[num])
        
        print('cost', Cost[num])
        
        Time_total[num] = Time_execute[num] + Time_match[num]
        Time_total_2[num] = results['time_total'] + Time_match[num]
        print('time_total', Time_total[num])
        print('time_total_2', Time_total_2[num])
        
        with fornax.Connection('sqlite:///mydb2.sqlite') as conn:  # introduce connection first 
            # cursor = conn.cursor()
            
            # cursor.execute('DELETE FROM graphs')
            query.delete()
            target_graph.delete()
            query_graph.delete()
            # fornax.conn.close()    
            # conn.commit()
            sql_statement = text(f'DELETE FROM match;')
            conn.session.execute(sql_statement)
            conn.session.commit()
            
            conn.close()
            
        #%%
        num += 1 
    
    #%%
    # Ratio_set.append(Ratio)
    Time_total_2_set.append(np.mean(Time_total_2))
    
    Index_set.append(Index)
    
    Rate_exact = sum(Index)/Num
    print('rate exact', Rate_exact)
    
    Rate_exact_set.append(Rate_exact)
    
    Cost_set.append(np.mean(Cost / N))
    
    index1 = [index for index, value in enumerate(Cost / N) if value < thre1]
    Rate1_subopt = len(index1) / Num
    Rate1_subopt_set.append(Rate1_subopt)
    
    index3 = [index for index, value in enumerate(Cost / N) if value < thre3]
    Rate3_subopt = len(index3) / Num
    Rate3_subopt_set.append(Rate3_subopt)
    
#%% 
# file_path_1 = "E:/Master Thesis/results/nema/Ratio.npy"
# file_path_2 = "E:/Master Thesis/results/nema/Time.npy"
# np.save(file_path_1, Ratio)
# np.save(file_path_2, Time)

# directory = "E:/Master Thesis/results/nema/local"
# file_path_1 = os.path.join(directory, "Ratio_noise_0.1_eps_"+str(Eps)+".npy")
# file_path_2 = os.path.join(directory, "Time_noise_0.1_eps_"+str(Eps)+".npy")
# file_path_3 = os.path.join(directory, "TimeMatch_noise_0.1_eps_"+str(Eps)+".npy")
# os.makedirs(directory, exist_ok=True)

# np.save(file_path_1, Ratio)
# np.save(file_path_2, Time)
# np.save(file_path_3, Time_match)

#%%
# print('the ratio of correct nodes', np.mean(Ratio))
# print('Time_execute', np.mean(Time_execute))
# print('Time_match', np.mean(Time_match))

# print('Time_opt', np.mean(Time_opt))
# print('Time_total', np.mean(Time_total))
# print('Time_total_new', np.mean(Time_total_2))

# print('cost_mean', np.mean(Cost / N))
# print('cost_std', np.std(Cost / N))

    #%%
    
        
        # # Create a cursor object to execute SQL statements
        # cursor = conn.cursor()
        
        # # Replace 'old_graph_id' with the current identifier of the graph you want to change
        # old_graph_id = target_graph.graph_id
        
        # # Replace 'new_graph_id' with the new identifier you want to set for the graph
        # new_graph_id = 0
        
        # # Execute the UPDATE statement for nodes table (replace 'nodes_table' with the actual table name)
        # cursor.execute("UPDATE nodes_table SET graph_id = ? WHERE graph_id = ?", (new_graph_id, old_graph_id))
        
        # # Execute the UPDATE statement for edges table (replace 'edges_table' with the actual table name)
        # cursor.execute("UPDATE edges_table SET graph_id = ? WHERE graph_id = ?", (new_graph_id, old_graph_id))
        
        # # Commit the changes to make the modification permanent
        # conn.commit()
        
        # # Close the cursor and the connection
        # cursor.close()
        # conn.close()