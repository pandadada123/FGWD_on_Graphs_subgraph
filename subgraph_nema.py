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

import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
import fornax

import random
import string

import numpy as np
from lib1.data_loader import load_local_data,histog,build_noisy_circular_graph


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import SVG

# Add project root dir
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)


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

# In[6]:
# dataset_n='mutag' 
# dataset_n='protein' 
# dataset_n='protein_notfull' 
# dataset_n='aids'
# dataset_n='ptc'   # color is not a tuple
# dataset_n='cox2'
dataset_n='bzr'

path='E:/Master Thesis/dataset/data/'
X,label=load_local_data(path,dataset_n,wl=0) 
NumG = len(X)

N=3 # size of query

# Generate a random string of given length
def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# # Create a list of 97 random strings
# target_labels = query_labels + [random_string(random.randint(1, 10)) for _ in range(97)]
# target_labels = query_labels + [random.choice(query_labels) for _ in range(100-5)]
# target_labels = ['guardians', 'groot', 'star', 'guardians2']

#%% Randomly select a graph
random_number = random.randint(0, NumG - 1) # randomly select a graph
random_number = 0
# g3 = random.choice(X).nx_graph
x = X[random_number].nx_graph

target_nodes = x.nodes() # [dict]
target_id = sorted(list(x.nodes.keys())) # [list] list of node_id
target_labels = []
for i in range(len(target_id)):
    key = target_id[i]
    target_labels.append(target_nodes[key]['attr_name'])
target_edges = list(x.edges())

#%% create connected query graphs by BFS
# Randomly select a graph
random_number = random.randint(0, NumG - 1) # randomly select a graph
random_number = 0
# g3 = random.choice(X).nx_graph
g3 = X[random_number].nx_graph
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

# query = subgraph
query_nodes = subgraph.nodes() # [dict]
query_id = sorted(list(subgraph.nodes.keys())) # [list] list of node_id; query_id has to be changed in fornax.Connection (always start from zero)
query_labels = []
for i in range(len(query_id)):
    key = query_id[i]
    query_labels.append(query_nodes[key]['attr_name'])
query_edges = list(subgraph.edges())

#%%
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

def node_scoring_function(f1, f2): # return 1 when two features are the same
     # fea_metric == 'sqeuclidean':        
    f1 = np.array(f1)
    f2 = np.array(f2)
    if f1.shape != f2.shape:
        return 0
    else: 
        return 1/(1+np.sum([pow(f1-f2 ,2)]))
    
# In[7]:
# def search(query_id: int, query_label: str):
def search(query_id, query_label):
    # compute all of the scores
    scores = nodes_df['target_label'].apply(
        node_scoring_function, 
        args=(query_label,)
    )
    
    # to include all labels in the candidate list
    small_value = 1e-20
    scores = scores.replace(0, small_value)
    
    # create a boolean mask
    epsilon = 1e-9
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
matches = pd.concat(search(id_, label) for id_, label in enumerate(query_labels))
matches


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


with fornax.Connection('sqlite:///mydb.sqlite') as conn:
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


with fornax.Connection('sqlite:///mydb.sqlite') as conn:
    target_graph.graph_id
    another_target_graph_handle = fornax.GraphHandle.read(conn, target_graph.graph_id)
    print(another_target_graph_handle == target_graph)


# ## Creating a query graph
# 
# Let's imagine that we suspect `groot` is directly related to `guardians` and `star` is also directly related to `guardians`.
# For example `groot` and `star` could both be members of a team called `guardians`.
# Let's create another small graph that represents this situation:

# In[12]:


with fornax.Connection('sqlite:///mydb.sqlite') as conn:
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


with fornax.Connection('sqlite:///mydb.sqlite') as conn:
    query = fornax.QueryHandle.create(conn, query_graph, target_graph)
    query.add_matches(matches['query_id'], matches['target_id'], matches['score'])
    
    # query.qeury_id = None


# Finally we can execute the query using a variety of options.
# We specify we want the top 5 best matches between the query graph and the target graph.

# In[14]:


with fornax.Connection('sqlite:///mydb.sqlite') as conn:
    # get_ipython().run_line_magic('time', 'results = query.execute(n=5)')
    results = query.execute(n=1, hopping_distance=1)  # top-n results

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

# In[16]:


for i, graph in enumerate(results['graphs'][:1]):
    plt.title('Result {0}, score: {1:.2f}'.format(1, 1. - graph['cost']))
    draw(graph)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.axis('off')
    plt.show()


# Results 2-4 have a lower score because `star` matches to a different node not adjacent to Guardians of the Galaxy. Further inspection would show that `star` has matched aliases of Star-Lord which are near Guardians of the Galaxy but not ajacent to it.

# In[17]:


for i, graph in enumerate(results['graphs'][1:4]):
    plt.title('Result {0}, score: {1:.2f}'.format(i+2, 1. - graph['cost']))
    draw(graph)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.axis('off')
    plt.show()


# The final match pairs `guardians` and `star` to two nodes that do not have similar edges to the target graph. `groot` is not found in the target graph. The result gets a much lower score than the preceding results and we can be sure that any additional results will also be poor because the result are ordered.

# In[18]:


for i, graph in enumerate(results['graphs'][4:]):
    plt.title('Result {0}, score: {1:.2f}'.format(i+5, 1. - graph['cost']))
    draw(graph)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.axis('off')
    plt.show()

#%%
# with fornax.Connection('sqlite:///mydb.sqlite') as conn:  # introduce connection first 
#     # query.delete()
#     # target_graph.delete()
#     # query_graph.delete()
#     fornax.cnn.close()
    
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