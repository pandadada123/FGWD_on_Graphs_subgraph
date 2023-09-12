# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib

# import torch
import ot
import lib0.optim as optim
from lib0.utils import dist,reshaper
# from bregman import sinkhorn_scaling
from scipy import stats
from scipy.sparse import random

class StopError(Exception):
    pass

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


def cal_L(C1,C2):  # loss_func is not specified
    """calculate the constant matrix L
    """
    L=np.zeros([len(C1),len(C1[1]),len(C2),len(C2[1])]) # L is a 4-dim tensor constant            
    for i in range(len(C1)):
        for ii in range(len(C1[1])):         
                for j in range(len(C2)-1):
                    for jj in range(len(C2[1])-1):   # a few elements are zero, related to the dummy node
                                   c1=C1[i][ii]
                                   c2=C2[j][jj]
                                    # L[i][ii][j][jj]=pow((c1-c2),2) 
                                   # L[i][ii][j][jj]=pow(_delta_plus(c1,c2) ,2) # only consider the case when c1>c2
                                   L[i][ii][j][jj]=_delta_plus(pow(0.5,c2),
                                                               pow(0.5,c1)) # only consider the case when c1>c2 == 0.5^c2>0.5^c1
                                   
    return L
    
    # Maybe another way to calculate L?                              
    # L_nodummy=np.zeros([len(C1),len(C1[1]),len(C2_nodummy),len(C2_nodummy[1])]) # L is a 4-dim tensor constant            
    # for i in range(len(C1)):
    #     for ii in range(len(C1[1])):         
    #             for j in range(len(C2_nodummy)):
    #                 for jj in range(len(C2_nodummy[1])):
    #                                c1=C1[i][ii]
    #                                c2=C2_nodummy[j][jj]
    #                                # np.append(L,pow((c1-c2),2))    
    #                                L_nodummy[i][ii][j][jj]=pow((c1-c2),2)  
    
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

    # tens=tensor_product(constC,hC1,hC2,T) 
    
    # return np.sum(tens*T) 1
    
    # return np.sum( tensor_matrix(L, T) * T)   + 0.1 * np.sum(np.log(T)*T) # the objective function of GW, entropic regularization
    # return np.sum( tensor_matrix(L, T) * T)   + 1 * np.sum(T*T)  # L2 norm
    return np.sum( tensor_matrix(L, T) * T) # original
    # a = 1001
    # return np.sum( tensor_matrix(L, T) * T) - 0.01 /(1-a) * np.log( np.sum (pow(T,a) ))

def gwggrad(L,T):
    
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
          
    # T_temp=np.delete(T,-1,axis=1)   # delete the last column of T
    # grad_temp=2*tensor_matrix(L_nodummy,T_temp)
    # grad = np.append( grad_temp, np.zeros([len(grad_temp),1]),  axis=1) # add a zero column
    
    # return grad
    
    # return 2*tensor_matrix(L, T)  + 0.1 * (  np.log(T)+np.ones(np.shape(T))  )
    # return 2*tensor_matrix(L, T)  + 1 * (  2*T  )
    return 2*tensor_matrix(L, T)
    # a = 100
    # return 2*tensor_matrix(L, T) - 0.01 /(1-a)/ np.sum (pow(T,a)) * a * pow(T,a-1)

def fgw_lp(M,C1,C2,C2_nodummy,p,q,q_nodummy,loss_fun='square_loss',alpha=1,amijo=True,G0=None,**kwargs): # alpha = 1 is default 
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
 
            
    
    # for GWD, M=0
                                
    if G0 is None:
        # G0=p[:,None]*q[None,:]
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
        
        G0 = np.outer(p, q) # ORIGINAL INITILAIZATION
        
        # G0 = np.ones ([len (p), len(q)])
        
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

        # temp=np.matlib.repmat(np.array([[1,-1],[-1,1]]), len(p)//2, len(q)//2)        
        # # epsilon = 2*1e-2 * np.random.randn()
        # epsilon =  1e-3*np.random.rand()
        # G0=G0+ np.append ( epsilon*temp , np.zeros([len(p),1]), axis = 1)
        
        # G0 = np.random.rand(len(C1),len(C2)) + 1e-5
        
    L=cal_L(C1,C2)
        
    def f(G):
        # return gwloss(constC,hC1,hC2,G)  # Return the Loss for Gromov-Wasserstein
        return gwloss(L,G)
    
    def df(G):
        
        # constC,hC1,hC2=init_matrix(C1,C2_nodummy,G,p,q_nodummy,loss_fun)  # q_nodummy is the distribution for calcualting the gradient; 
                                                                    # const,hC1,hC2 are only for calculating the gradient
        # return gwggrad(L_nodummy,G)  # Return the gradient for Gromov-Wasserstein
        return gwggrad(L,G)
    
    # return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=constC,**kwargs) # for GWD, alpha = reg=1, M=0
    return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=None,**kwargs) # for GWD, alpha = reg=1, M=0
    # 
    #%%
    # return ot.gromov.entropic_gromov_wasserstein(C1=C1, C2=C2, p=p, q=q, loss_fun=f, epsilon=0.01,
    #                             max_iter=1000, tol=1e-9, verbose=False, log=False)
