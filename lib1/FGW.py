# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib

# import torch
import ot
import lib1.optim as optim
from lib1.utils import dist,reshaper
# from bregman import sinkhorn_scaling
from scipy import stats
from scipy.sparse import random

class StopError(Exception):
    pass

""" trick by Peyre 2016 and Liu 2023"""

def init_matrix(C1,C2,T,p,q,loss_fun='square_loss'):
    """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [1]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    .. [2] X. Liu, R. Zeira, and B. J. Raphael, 
    “PASTE2: Partial Alignment of Multi-slice Spatially Resolved Transcriptomics Data.” 
    bioRxiv, p. 2023.01.08.523162, Jan. 08, 2023. doi: 10.1101/2023.01.08.523162.

    """
    
    if loss_fun == 'square_loss':
        def f1(a):
            return a**2 

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2*b
    
    elif loss_fun == 'kl_loss': # from Liu 2023
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)
        
    #%%
    # constC1 = np.dot(
    #                  np.dot(f1(C1), p.reshape(-1, 1)),
    #                  np.ones(len(q)).reshape(1, -1)
    #                  )
    # constC2 = np.dot(
    #                  np.ones(len(p)).reshape(-1, 1),
    #                  np.dot(q.reshape(1, -1), f2(C2).T)
    #                  )
    
    #%% also suitable for partial coupling (marginals of T does not sum up to 1) from Liu 2023
    ## T matrix is also the input of this function
    ## p and q are actually no need 
    constC1 = np.dot(
                    f1(C1),
                    np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1))
                    )

    constC2 = np.dot(
                    np.dot(np.ones(C1.shape[0]).reshape(1, -1), T),
                    f2(C2).T
                    )
     #%% also suitable for partial coupling (marginals of T does not sum up to 1) from Liu 2023
    # ## SAME RESULTS AS ABOVE
    # ## T matrix is also the input of this function
    # ## p and q are actually no need 
    # constC1 = np.dot(
    #                 f1(C1),
    #                 np.dot(
    #                        T, 
    #                        np.dot(np.ones(C2.shape[0]).reshape(-1, 1) , np.ones(C2.shape[0]).reshape(1,-1) )
    #                        )
    #                 )

    # constC2 = np.dot(
    #                 np.dot(
    #                        np.dot(np.ones(C1.shape[0]).reshape(-1, 1) ,  np.ones(C1.shape[0]).reshape(1, -1) ), 
    #                        T
    #                        ),
    #                 f2(C2).T
    #                 )
    
    #%%
    constC=constC1+constC2
    hC1 = h1(C1)
    hC2 = h2(C2)
        
    return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):

    """ Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    
    A=-np.dot(hC1, T).dot(hC2.T)
    tens = constC+A

    return tens

def gwloss(constC,hC1,hC2,T):

    """ Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    
    T_nodummy = T[:,0:len(T[0])-1]  
    
    tens=tensor_product(constC,hC1,hC2,T_nodummy) 
    
    tens=np.append(tens,  np.zeros([len(tens),1]),  axis=1)  # add a zero column 
               
    return np.sum(tens*T) 
    # return np.sum( tens * T)   + 0.01 * np.sum(np.log(T)*T)

def gwggrad(constC,hC1,hC2,T):
    
    """ Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    
    T_nodummy = T[:,0:len(T[0])-1]  
    
    tens=tensor_product(constC,hC1,hC2,T_nodummy) 
    
    tens=np.append(tens,  np.zeros([len(tens),1]),  axis=1)  # add a zero column 
    
    return 2*tens
    # return 2*tens+ 0.01 * (  np.log(T)+np.ones(np.shape(T))  )

def fgw_lp(M,C1,C2,C2_nodummy,p,q,q_nodummy,loss_fun='square_loss',alpha=1,amijo=True,G0=None,stopThr=1e-09,**kwargs): 
    """
    Computes the FGW distance between two graphs see [3]
    .. math::
        \gamma = arg\min_\gamma (1-\alpha)*<\gamma,M>_F + alpha* \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    M  : ndarray, shape (ns, nt)
         Metric cost matrix between features across domains
         IF ALPHA=1, THEN M IS ZERO MATRIX
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix respresentative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
         Metric cost matrix espresentative of the structure in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string,optionnal
        loss function used for the solver 
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    amijo : bool, optional
        If True the steps of the line-search is found via an amijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly pased to the ot.optim.cg solver
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    
    # constC,hC1,hC2=init_matrix(C1,C2_nodummy,p,q_nodummy,loss_fun)        
    
    # for GWD, M=0
                     
    if G0 is None:
        
   #%% G0 initialization 
        G0=p[:,None]*q[None,:]
        # G0=np.outer(ot.unif(10),ot.unif(6))
        # G0 = np.outer(np.array([0.05,0.05,0.05,0.05,0.05, # not right, does not satisfy the constraint
        #                         0.05,0.05,0.05,0.05,0.55]),np.array([1/6,1/6,1/6,1/6,1/6,1/6]))
        
        # G0=np.random.randn(9,5)*0.02
        # G0=abs(G0)
        # t0=G0.sum(axis=0)
        # tt0=q[0:len(q)-1]-t0
        # tt0=tt0[None,...]
        # G0=np.r_[G0,tt0]
        # t1=G0.sum(axis=1)
        # G0=np.c_[G0,p-t1]
        
        # G0=abs(G0)
        
        # G0 = np.outer(p, q)
        
        # G0[:-1] = 0 # set last column to be zero 
        # G0[:-1] = 100
        
        # G0 = np.zeros([10,6])
        # G0[0,0]=1 # This works very well!?
        
        # G0[0,0]=G0[0,0]-0.008
        # G0[0,-1]=G0[0,-1]+0.008
        # G0[-1,-1]=G0[-1,-1]-0.008
        # G0[-1,0]=G0[-1,0]+0.008
        
        # G0=np.array([[0.1,0,0,0,0,0],[0,0.1,0,0,0,0],[0,0,0.1,0,0,0],[0,0,0,0.1,0,0],[0,0,0,0,0.1,0],
        #             [0,0,0,0,0,0.1],[0,0,0,0,0,0.1],[0,0,0,0,0,0.1],[0,0,0,0,0,0.1],[0,0,0,0,0,0.1]])

        # temp=np.matlib.repmat(np.array([[1,-1],[-1,1]]), 5, 3)        
        # epsilon = 2*1e-2 * np.random.randn()
        # G0=G0+epsilon*temp 
        
        # np.random.seed()
        # G0 = np.random.normal(0.1, 0.01, size = (len(C1),len(C2)) )
        # G0 = np.random.rand(len(C1),len(C2))
        # print(G0)
        
   #%% no trick     
    # L=cal_L(C1,C2)
    
    # def f(G):
    #     return gwloss(L,G)
    
    # def df(G):        
    #     return gwggrad(L,G)
    # return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=None,**kwargs) # for GWD, alpha = reg=1, M=0

   #%% trick by Peyre and partial aligment
    def f(G):  # objective function of GWD
        G_nodummy = G[:,0:len(G[0])-1]  
        constC,hC1,hC2=init_matrix(C1,C2_nodummy,G_nodummy,p,q_nodummy,loss_fun)        

        return gwloss(constC,hC1,hC2,G)
    
    def df(G): # gradient of GWD
        G_nodummy = G[:,0:len(G[0])-1]  
        constC,hC1,hC2=init_matrix(C1,C2_nodummy,G_nodummy,p,q_nodummy,loss_fun)        

        return gwggrad(constC,hC1,hC2,G)
    
    # return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=constC,**kwargs) # for GWD, alpha = reg=1, M=0
    # constC is not used in optim process
    return optim.cg(p,q,M,alpha,f,df,G0,stopThr=stopThr,amijo=amijo,C1=C1,C2=C2,constC=None,**kwargs) # for GWD, alpha = reg=1, M=0 
     