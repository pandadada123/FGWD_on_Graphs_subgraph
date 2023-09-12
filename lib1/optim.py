# -*- coding: utf-8 -*-
"""
Optimization algoriterhms for OT
"""


import numpy as np
# from scipy.optimize.linesearch import scalar_search_armijo
from ot.lp import emd


class StopError(Exception):
    pass


class NonConvergenceError(Exception):
    pass
class StopError(Exception):
    pass

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
    alpha > 0 is assumed to be a descent direction.
    Returns
    -------
    alpha
    phi1
    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0: # This is the formula for amijo backtracking
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1
        
def do_linesearch(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=0.99): # amijo
    """
    Armijo linesearch function that works with matrices
    find an approximate minimum of f(xk+alpha*pk) that satifies the
    armijo conditions.
    Parameters
    ----------
    f : function
        loss function/objective function
    xk : np.ndarray
        initial position (The position that we are focusing on.)
    pk : np.ndarray
        descent direction
    gfk : np.ndarray
        gradient of f at xk
    old_fval : float
        loss value of f at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
        [c1 is the constant alpha in my notes]
    alpha0 : float, optional
        initial step (>0)
    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call --> f_count
    fa : float
        loss value at step alpha
    """
    xk = np.atleast_1d(xk)  # converted into array
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args) # function value after one step

    if old_fval is None:
        phi0 = phi(0.)   # initialization: phi(0)=f(xk)
    else:
        phi0 = old_fval

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices (Frob inner product)   
                                # derphi0 is the inner product of descent direction and gradient,
                                # which will be used in backtracking (armijo).
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)
                                # phi1: function value after this step (step size is alpha)

    return alpha, fc[0], phi1

def cg(a, b, M, reg, f, df, G0=None, numItermax=500, stopThr=1e-09, verbose=False,log=False,amijo=True,C1=None,C2=None,constC=None):
    """
    Solve the general regularized OT problem witerh conditerional gradient
        The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg*f(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is iters gradient)
    - a and b are source and target weights (sum to 1)
    The algoriterhm used for solving the problem is conditerional gradient as discussed in  [1]_
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix [for Wasserstein]
    reg : float
        Regularization term >0
    G0 :  np.ndarray (ns,nt), optional
        initerial guess (default is indep joint densitery)
    numiterermax : int, optional
        Max number of itererations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along itererations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """

    loop = 1

    if log:
        log = {'loss': [],'delta_fval': [],'gradient':[], 'descent':[],'G':[],'alpha':[], 'Gc':[]}

    if G0 is None:
        G = np.outer(a, b)   # use "outer" to initialize G=G0
    else:
        G = G0
    
    n = len(a)
    m = len(b)
    def cost(G):  # objective of FGWD
        obj = np.sum(M * G) / (m/n) + reg * f(G) / (pow(m,2)/pow(n,2))
        
        # obj = np.sum(M * G) + reg * f(G) # original FGWD
        # obj = obj / ((1-reg)*m/n + reg*pow(m,2)/pow(n,2)) # normalized FGWD
        return obj  # (Regularized discrete optimal transport)
                                            # for GWD, M is zero matrix. and reg=1
                                            # M is actually already (1-alpha)M for FGWD
                                            
    f_val = cost(G) #f(xt)  initerialization of f_val

    if log:
        log['loss'].append(f_val)
        log['G'].append(G)

    iter = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'iter.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(iter, f_val, 0))

    while loop:

        iter += 1
        old_fval = f_val # update objective function
        #G=xt
        # problem linearization
        # Mi = M + reg * df(G) #Gradient(xt)
        # G_shape=np.shape(G)
        # G2=G[:,0:G_shape[1]-1]  
        # G3=df(G2) # gradient calculated by the original method 
        # G4=np.append(G3,  np.zeros([G_shape[0],1]),  axis=1)
        # G4=np.append(G3,  np.ones([G_shape[0],1])*100,  axis=1)
        
        Mi = M + reg * df(G) #Gradient(xt) of FGWD
        # Mi = M + reg * G4 #Gradient(xt) For Gromov, M=0, Mi=G4
                            # For FGWD, M is the gradient of WD and df(G) is of GWD.

        # set M positive 
        Mi += Mi.min()  # no big effects 
        # Mi-= Mi.min() 
        # Mi += np.min(Mi) # is the same as original one

        # solve linear program (Wasserstein is LP)
        Gc = emd(a, b, Mi) #st  actually same as EMD problem, minimizeg the inner product to find the argmin(Gc), has to satisfy the constraints of a and b
                            # 只是利用EMD的结构特点计算 Also within the constraints
        deltaG = Gc - G # dt:deltaG : ndarray (ns,nt) Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration  
                        # Gc is the argmin matrix and G is the old one
                        # This is the descent direction = pk
                        # deltaG: descent direction; Mi: gradient
                        
        # argmin_alpha f(xt+alpha dt)  # find the step size alpha # Frank-Wolfe with linesearch?
        alpha, fc, f_val = do_linesearch(f=cost,xk=G,pk=deltaG,gfk=Mi,old_fval=f_val,
                                          args=(), c1=1e-4, alpha0=0.99)
        
        # alpha = 2/(iter+2)
        
        if alpha is None or np.isnan(alpha):
            raise NonConvergenceError('Alpha was not found')
            # alpha = 0.99
            # alpha = 2/(iter+2)
        else:
            G = G + alpha * deltaG #xt+1=xt +alpha dt

        # test convergence
        if iter >= numItermax:
            loop = 0
            
        delta_fval = (f_val - old_fval)

        #delta_fval = (f_val - old_fval)/ abs(f_val)
        #print(delta_fval)
        if abs(delta_fval) < stopThr:    # check if can stop
            loop = 0

        if log:  # add new results to log
            log['loss'].append(f_val)
            log['delta_fval'].append(delta_fval)
            log['gradient'].append(Mi)
            log['descent'].append(deltaG)
            log['G'].append(G)
            log['alpha'].append(alpha)
            log['Gc'].append(Gc)

        if verbose:
            if iter % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'iter.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}|{:5e}'.format(iter, f_val, delta_fval,alpha))

    if log:
        return G, log
    else:
        return G
