import ot
import FGW as fgw
import numpy as np
import time
from graph import NoAttrMatrix
from utils import hamming_dist
import networkx as nx
import sys
import fornax as fn
"""
The following classes adapt the OT distances to Graph objects
"""

class BadParameters(Exception):
    pass

# class Wasserstein_distance():
#     """ Wasserstein_distance is a class used to compute the Wasserstein distance between features of the graphs.
    
#     Attributes
#     ----------    
#     features_metric : string
#                       The name of the method used to compute the cost matrix between the features
#     transp : ndarray, shape (ns,nt) 
#            The transport matrix between the source distribution and the target distribution 
#     """
#     def __init__(self,features_metric='sqeuclidean'): #remplacer method par distance_method  
#         self.features_metric=features_metric
#         self.transp=None

#     def reshaper(self,x):
#         x=np.array(x)
#         try:
#             a=x.shape[1]
#             return x
#         except IndexError:
#             return x.reshape(-1,1)

#     def graph_d(self,graph1,graph2,t1masses,t2masses):
#         """ Compute the Wasserstein distance between two graphs. Uniform weights are used.        
#         Parameters
#         ----------
#         graph1 : a Graph object
#         graph2 : a Graph object
#         Returns
#         -------
#         The Wasserstein distance between the features of graph1 and graph2
#         """

#         nodes1=graph1.nodes()  # nodes of two graphs
#         nodes2=graph2.nodes()
#         # t1masses = np.ones(len(nodes1))/len(nodes1)  # uniform weights
#         # t2masses = np.ones(len(nodes2))/len(nodes2)
#         x1=self.reshaper(graph1.all_matrix_attr())
#         x2=self.reshaper(graph2.all_matrix_attr())

#         if self.features_metric=='dirac':
#             # f=lambda x,y: x!=y   # x not equal to y
#             f=lambda x,y: int(x!=y)   # x not equal to y
#             M=ot.dist(x1,x2,metric=f)
#         else:
#             M=ot.dist(x1,x2,metric=self.features_metric) 
#         if np.max(M)!=0:
#             M= M/np.max(M)
#         self.M=M

#         transp = ot.emd(t1masses,t2masses, M)
#         self.transp=transp

#         return np.sum(transp*M), transp

#     def get_tuning_params(self):
#         return {"features_metric":self.features_metric}



class Prox():
    """ Fused_Gromov_Wasserstein_distance is a class used to compute the Fused Gromov-Wasserstein distance between graphs 
    as presented in [3]
    
    Attributes
    ----------  
    alpha : float 
            The alpha parameter of FGW
    method : string
             The name of the method used to compute the structures matrices of the graphs. See Graph class
    max_iter : integer
               Number of iteration of the FW algorithm for the computation of FGW.
    features_metric : string
                      The name of the method used to compute the cost matrix between the features
                      For hamming_dist see experimental setup in [3]
    transp : ndarray, shape (ns,nt) 
           The transport matrix between the source distribution and the target distribution
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.  
            If there is convergence issues use False.
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    def __init__(self,alpha=0.5,method='shortest_path',features_metric='sqeuclidean',loss_fun ='square_loss', max_iter=500,verbose=False,amijo=True): #remplacer method par distance_method  
        self.method=method
        self.max_iter=max_iter
        self.alpha=alpha
        self.features_metric=features_metric
        self.transp=None
        self.log=None
        self.verbose=verbose
        self.amijo=amijo
        #if alpha==0 or alpha==1:
        #    self.amijo=True
        self.loss_fun=loss_fun

    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def calc_fgw(self,M,C1,C2,C2_nodummy,t1masses,t2masses,p2_nodummy):
        transpwgw,log= fgw.fgw_lp((1-self.alpha)*M,C1,C2,C2_nodummy,t1masses,t2masses,p2_nodummy,self.loss_fun,G0=None,alpha=self.alpha,verbose=self.verbose,amijo=self.amijo,log=True)      
        return transpwgw,log
        
    def graph_d(self,graph1,graph2,t1masses,t2masses,p2_nodummy):
        """ Compute the Fused Gromov-Wasserstein distance between two graphs. Uniform weights are used.        
        Parameters
        ----------
        graph1 : a Graph object
        graph2 : a Graph object
        Returns
        -------
        The Fused Gromov-Wasserstein distance between the features of graph1 and graph2
        """
        gofeature=True
        
        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        startstruct=time.time()
        
        n1=len(nodes1)
        n2=len(nodes2)
        
        # Keys1 = sorted(list(nodes1.keys()))  # order of nodes for cost matrices
        # Keys2 = sorted(list(nodes2.keys()))
        Keys1 = list(nodes1.keys()) # order of nodes for cost matrices (DO NOT NEED TO BE SORTED)
        Keys2 = list(nodes2.keys())
        
        #%% Calculate the shortest path distance matrix 
        # LargeValue = sys.float_info.max
        LargeValue = 1e6
        
        # def shortest(G):
        #     n = len (G.nodes())
        #     C = np.zeros ([n,n])
        #     for i in range(n):
        #         for ii in range(n):  
        #             try:
        #                 C[i][ii]=nx.shortest_path_length(G.nx_graph,source=i,target=ii)
        #             except: 
        #                 C[i][ii]=LargeValue
        #     return C
        
        def shortest(G,Keys):
            g = G.nx_graph
            # Nodes = list(g.nodes())  # list of nodes/keys, no sort
            # Keys = sorted(G.nodes().keys())
            # n = len (G.nodes())
            n = len(Keys)
            C = np.zeros([n,n])
            for i in range(n):
                for ii in range(n):
                    try:
                        C[i][ii]=nx.shortest_path_length(g,source=Keys[i],target=Keys[ii])
                    except: 
                        C[i][ii]=LargeValue
            return C
        
        def adj(G,Keys):
            C = nx.to_numpy_array(G.nx_graph, nodelist=Keys)
            
            return C
            
        def _proximity(h: float, Alpha: float, distances: np.ndarray) -> np.ndarray:
            """Calculates the proximity factor P for an array of distances.
            Implements equation 1 in the paper
        
            Arguments:
                h {float} -- max hopping distance
                alpha {float} -- propagation factor
                distances {np.array} -- array of hopping distances
        
            Raises:
                ValueError -- if hopping distance is less than zero
                ValueError -- if propagation factor is not between zero and one
        
            Returns:
                np.array -- an array of proximiy values
            """
        
            if h < 0:
                raise ValueError('hopping distance h cannot be negative')
            if not 0 < Alpha <= 1:
                raise ValueError('propagation factor alpha must be between 0 and 1')
            return np.multiply(
                np.less_equal(distances, h),
                np.power(Alpha, distances)
            )
        
        
        def _delta_plus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Comparator function. Equation 3 in the paper.
        
            Arguments:
                x {np.array} -- an array of floats
                y {np.array} -- an array of floats
        
            Returns
                np.array -- an array of floats
            """
        
            return np.multiply(
                np.greater(x, y),
                np.subtract(x, y)
            )
        
        # def Proximity(G,Keys,h=1,Alpha=0.5):
        #     D=shortest(G, Keys)
        #     # n = len(Keys)
        #     Nei = _proximity(h=h,Alpha = Alpha, distances = D)
        #     return Nei
        
        # def find_h_hop_nodes(graph, source_node, h): # h=1 is the same as the adj
        #     # Create an empty set to store the h-hop nodes
        #     h_hop_nodes = list()
        
        #     # Use NetworkX's breadth-first search to explore the graph up to h hops
        #     bfs_edges = nx.bfs_edges(graph, source_node)
        
        #     for u, v in bfs_edges:
        #         # Check if the distance from the source node to the current node is less than or equal to h
        #         if nx.shortest_path_length(graph, source_node, v) <= h:
        #             # Add the current node to the set of h-hop nodes
        #             h_hop_nodes.append(v)
        
        #     return h_hop_nodes

        def find_h_hop_nodes(index, distance, h): # h=1 is the same as the adj
            # Create an empty set to store the h-hop nodes
            h_hop_nodes = list()
            
            n=distance.shape[-1]
            for index2 in range(n):
                # Check if the distance from the source node to the current node is less than or equal to h
                if distance[index,index2] <= h:
                    # Add the current node index to the set of h-hop nodes
                    h_hop_nodes.append(index2)
        
            return h_hop_nodes
        
        def NeiCost(G1,Keys1,G2,Keys2, h=1, Alpha = 0.5):
            g1 = G1.nx_graph
            g2 = G2.nx_graph
            D1=shortest(G1, Keys1)
            D2=shortest(G2, Keys2)
            
            # Nei1 = Proximity(G1,Keys1,h=1,Alpha=0.5)
            # Nei2 = Proximity(G2,Keys2,h=1,Alpha=0.5)
            Nei1 = _proximity(h=h, Alpha = Alpha, distances = D1)
            Nei2 = _proximity(h=h, Alpha = Alpha, distances = D2)
            
            C = np.zeros([n1,n2])
            for i in range(n1):
                for j in range(n2):
                    # k1=Keys1[i]
                    # k2=Keys2[j]
                    hop1 = find_h_hop_nodes(i, D1, h)
                    hop2 = find_h_hop_nodes(j, D2, h)
                    
                    temp=0
                    for uu in hop1:
                        for vv in hop2:                                                  
                            # temp += _delta_plus(Nei2[j, Keys2.index(vv)],
                            #                     Nei1[i, Keys1.index(uu)])  # without normalization; query graph is the first variable 
                            temp += _delta_plus(Nei2[j, vv],
                                                Nei1[i, uu])  # without normalization; query graph is the first variable 
                    
                    # # find the indexes of elements in hop1, in Keys1
                    # List1=[index for index, element in enumerate(Keys1) if element in hop1]
                    # List2=[index for index, element in enumerate(Keys2) if element in hop2]
                    
                    # temp=_delta_plus(Nei1[i,List1], 
                    #                  Nei2[j,List2]) # np.array                 
                    # C[i][j]=np.sum(temp)
                    
                    C[i][j] = temp
                    
            return C
        
        # C1=graph1.distance_matrix(method=self.method)
        # C1 = np.zeros ([n1,n1])
        # for i in range(n1):
        #     for ii in range(n1):  
        #         C1[i][ii] = nx.shortest_path_length(graph1.nx_graph,source=i,target=ii)
                
        
        # C2=graph2.distance_matrix(method=self.method)
        # C2_nodummy = np.zeros ([n2-1,n2-1])
        # for j in range(n2-1):
        #     for jj in range(n2-1):  
        #         C2_nodummy[j][jj] = nx.shortest_path_length(graph2.nx_graph,source=j,target=jj)
        
        # C2_shape=np.shape(C2)        
        # C2_nodummy=C2[:,0:C2_shape[1]-1]
        # C2_nodummy=C2_nodummy[0:C2_shape[0]-1,:]
        
        # C2_temp=np.append(C2_nodummy,100*np.ones([n2-1,1]),axis=1)
        # C2=np.append(C2_temp,100*np.ones([1,n2]),axis=0)
        # C2[n2-1,n2-1]=0

        if self.method=='shortest_path':
            C1 = shortest(graph1,Keys1)
            C2 = shortest(graph2,Keys2)
        
        elif self.method=='adj':
            # Use adjacency matrices 
            # gg1 = graph1.nx_graph
            # gg2 = graph2.nx_graph
            # C1 = nx.to_numpy_array(gg1,nodelist=(list(gg1._node.keys())).sort())
            # C2 = nx.to_numpy_array(gg2,nodelist=(list(gg2._node.keys())).sort())
            # C1 = nx.to_numpy_array(gg1)
            # C2 = nx.to_numpy_array(gg2)
            C1 = adj(graph1,Keys1)
            C2 = adj(graph2,Keys2)
        
        elif self.method=='proximity':
            C = NeiCost(graph1,Keys1,graph2,Keys2,h=5)
            C[:,-1] = 0 # set the last col to be 0
            
                        
        # else:
        #     C1 = np.zeros([n1,n1])
        #     C2 = np.zeros([n2,n2])
            
        # C2_nodummy=C2[0:n2-1,0:n2-1]
        
        #%%
        end2=time.time()
        # t1masses = np.ones(len(nodes1))/len(nodes1)
        # t2masses = np.ones(len(nodes2))/len(nodes2)  # uniform weights
        # try :
        #     x1=self.reshaper(graph1.all_matrix_attr())
        #     x2=self.reshaper(graph2.all_matrix_attr())
        # except NoAttrMatrix:
        #     x1=None
        #     x2=None
        #     gofeature=False
        
        
        if gofeature : 
            M=np.zeros((n1,n2)) # initialization
            
            if self.features_metric=='dirac': # (does not work) Dirac is the same as hamming_dist for scalar comparison
                # f=lambda x,y: x!=y
                # f=lambda x,y: int(x!=y)
                # M=ot.dist(x1,x2,metric=f)
                
                for i in range(n1):
                    for j in range(n2):    
                        key1 = Keys1[i]
                        key2 = Keys2[j] 
                        f1 = nodes1[key1]['attr_name']
                        f2 = nodes2[key2]['attr_name']
                        if f1==f2:
                            M[i,j]=0
                        else:
                            M[i,j]=1
                    
            elif self.features_metric=='hamming': #see experimental setup in the original paper
                # f=lambda x,y: hamming_dist(x,y)
                # M=ot.dist(x1,x2,metric=f)
                # M=ot.dist(x1,x2,metric='hamming')
                for i in range(n1):
                    for j in range(n2):
                        key1 = Keys1[i]
                        key2 = Keys2[j] 
                        f1 = np.array( nodes1[key1]['attr_name'] )
                        f2 = np.array( nodes2[key2]['attr_name'] )
                        if f1.shape != f2.shape:
                            M[i,j] = LargeValue
                        else: 
                            M[i,j] = np.count_nonzero(f1 != f2)
                            
            elif self.features_metric=='sqeuclidean':
                for i in range(n1):
                    for j in range(n2):
                        key1 = Keys1[i]
                        key2 = Keys2[j] 
                        f1 = np.array( nodes1[key1]['attr_name'] )
                        f2 = np.array( nodes2[key2]['attr_name'] )
                        if f1.shape != f2.shape:
                            M[i,j] = LargeValue
                        else: 
                            M[i][j]=sum(pow(f1-f2 ,2))
                        
                
            # else:
                # M=ot.dist(x1,x2,metric=self.features_metric)
                
            # self.M=M
        else:
            M=np.zeros((C1.shape[0],C2.shape[0]))

        M[:,-1] = 0 # set the last col to be 0

#%%
        # startdist=time.time()
        # transpwgw,log=self.calc_fgw(M,C1,C2,C2_nodummy,t1masses,t2masses,p2_nodummy) #return the transport matrix and FGWD value
        # transpwgw,log=self.calc_fgw(M,C,t1masses,t2masses,p2_nodummy) #return the transport matrix and FGWD value
        Cost = (1-self.alpha)*M+self.alpha*C
        transp = ot.emd(t1masses,t2masses, Cost )
        # enddist=time.time()
        print(transp)
        
        # log['struct_time']=(end2-startstruct)
        # log['dist_time']=(enddist-startdist)
        # self.transp=transpwgw
        # self.log=log

        # return log['loss','delta_fval','gradient', 'descent'][::-1][0],transpwgw  # the loss value in the last iteration is the returned value
        # return log['loss'][::-1][0],log,transpwgw,M,C
        return transp,M,C

    def get_tuning_params(self):
        """Parameters that defined the FGW distance """
        return {"method":self.method,"max_iter":self.max_iter,"alpha":self.alpha,
        "features_metric":self.features_metric,"amijo":self.amijo}