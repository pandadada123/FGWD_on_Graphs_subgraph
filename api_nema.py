# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:24:03 2023

@author: Pandadada
"""

import opt_nema as opt

# %% create candidate list based on feature costs (but we do not need this)
# feature cost for strings
def node_scoring_function(first: str, second: str):  
    """ node scoring function takes two strings and returns a 
        score in the range 0 <= score <= 1
    """
    first_, second_ = sorted((first.lower(), second.lower()), key=len)
    # if first is not a substring of second: score = 0
    if not first_ in second_:
        return 0
    # otherwise use the relative difference between
    # the two lengths
    score = len(second_) - len(first_)
    score /= max(len(first_), len(second_))
    score = 1. - score
    return score

#%%
def _optimise(self, hopping_distance, max_iters, offsets):
    packed = opt.solve(
        records,
        hopping_distance=hopping_distance,
        max_iters=max_iters
    )
    inference_costs, subgraphs, iters, sz, target_edges_arr = packed
    return inference_costs, subgraphs, iters, sz, target_edges_arr

@classmethod
def _get_scores(cls, inference_costs, query_nodes, subgraphs, sz):
    scores = []
    for subgraph in subgraphs:
        score = sum(inference_costs[k] for k in subgraph)
        score += sz - len(subgraph)
        score /= len(query_nodes)
        scores.append(score)
    return scores

def execute(G1, G2, n=5, hopping_distance=2, max_iters=10):
    """Execute a fuzzy subgraph matching query finding the top *n* subgraph
    matches between the query graph and the target graph.

    :param n: number of subgraph matches to return
    :type n: int, optional
    :param hopping_distance: lengthscale hyperparameter, defaults to 2
    :type hopping_distance: int, optional
    :param max_iters: maximum number of optimisation iterations
    :type max_iters: int, optional
    :return: query result
    :rtype: dict
    """

    # offsets = None  # TODO: implement batching
    # self._check_exists()
    # if not len(self):
    #     raise ValueError('Cannot execute query with no matches')

    # graphs = []
    query_nodes = sorted(self._query_nodes())
    target_nodes = sorted(self._target_nodes())
    # we will with get target edges from the optimiser
    # since the optimiser knows this anyway
    target_edges = None
    query_edges = sorted(self._query_edges())

    packed = self._optimise(hopping_distance, max_iters, offsets)
    
    inference_costs, subgraphs, iters, sz, target_edges_arr = packed
    
    target_edges = self._target_edges(target_nodes, target_edges_arr)
    target_edges = sorted(target_edges)

    scores = self._get_scores(inference_costs, query_nodes, subgraphs, sz)
    
    # sort graphs by score then deturministicly by hashing
    idxs = sorted(
        enumerate(scores),
        key=lambda x: (x[1], self.conn._hash(tuple(subgraphs[x[0]])))
    )


    return {
        'graphs': graphs,
        'iters': iters,
        'hopping_distance': hopping_distance,
        'max_iters': max_iters
    }