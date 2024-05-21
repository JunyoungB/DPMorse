import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import scipy as cp
from scipy.optimize import root
from scipy.sparse import csgraph as cg
from scipy.integrate import solve_ivp
from scipy.cluster import hierarchy


def pred_density_score(x, pi, m, inv_sigma, coef):
    delta = np.array([(x-m[k])@inv_sigma[k]@(x-m[k]) for k in range(len(m))])
    den = pi*np.exp(-0.5*delta)*coef
    return den

def pred_density_score_mult(x, pi, m, inv_sigma, coef):
    delta = np.array([np.sum((x-m[k])@inv_sigma[k]*(x-m[k]),axis=1) for k in range(len(m))]).T
    den = pi*np.exp(-0.5*delta)*coef
    return den

def pred_density(x, pi, m, inv_sigma, coef):
    delta = np.array([(x-m[k])@inv_sigma[k]@(x-m[k]) for k in range(len(m))])
    den = np.sum(pi*np.exp(-0.5*delta)*coef)
    return -den

def pred_density_mult(x, pi, m, inv_sigma, coef):
    delta = np.array([np.sum((x-m[k])@inv_sigma[k]*(x-m[k]),axis=1) for k in range(len(m))]).T
    den = np.sum(pi*np.exp(-0.5*delta)*coef, axis=1)
    return -den

def pred_grad(x, pi, m, inv_sigma, coef):
    delta = np.array([(x-m[k])@inv_sigma[k]@(x-m[k]) for k in range(len(m))])
    sigma_mu = np.array([inv_sigma[k]@(x-m[k]) for k in range(len(m))])   
    return np.sum((pi*coef*np.exp(-0.5*delta)).reshape(-1,1)*sigma_mu, axis=0)

def pred_grad_with_t(t, x, pi, m, inv_sigma, coef):
    delta = np.array([(x-m[k])@inv_sigma[k]@(x-m[k]) for k in range(len(m))])
    sigma_mu = np.array([inv_sigma[k]@(x-m[k]) for k in range(len(m))])   
    grad = -np.sum((pi*coef*np.exp(-0.5*delta)).reshape(-1,1)*sigma_mu, axis=0)
    return grad/(1+np.linalg.norm(grad))

def pred_hess(x, pi, m, inv_sigma, coef):
    delta = np.array([(x-m[k])@inv_sigma[k]@(x-m[k]) for k in range(len(m))])
    sigma_2_mu_2 = np.array([np.outer(inv_sigma[k]@(x-m[k]),inv_sigma[k]@(x-m[k])) for k in range(len(m))])
    h = (pi*coef*np.exp(-0.5*delta)).reshape(-1,1,1)*sigma_2_mu_2
    h -= (pi*coef*np.exp(-0.5*delta)).reshape(-1,1,1)*inv_sigma
    return -1*np.sum(h, axis=0)

def findTPs(pi, m, inv_sigma, coef):
    N, p = m.shape
    ts = {}
    ts['x'] = []
    ts['f'] = []
    ts['neighbor'] = []
    ts['purturb'] = []
    tmp_x = []
    epsilon = 0.05
    for i in range(N):
        for j in range(i+1, N):
            seg = (m[j]-m[i])*(1+np.arange(20)).reshape(-1,1)/21 + m[i]
            tmp = seg[np.argmax(pred_density_mult(seg, pi, m, inv_sigma, coef))]
            res = solve_ivp(fun=lambda t, y : pred_grad_with_t(t, y, pi, m, inv_sigma, coef), t_span=[0,0.1], y0 = tmp)
            m_tmp = res.y.T[1]
            sol = root(fun=pred_grad, x0=m_tmp, jac = pred_hess, args = (pi, m, inv_sigma, coef))
            tmp_x.append(sol.x)

            for _ in range(5):
                u = m[j] - m[i]
                v = m_tmp - m[i]
                seg_u = (1+np.arange(20)).reshape(-1,1) / 21 
                seg_v = - seg_u ** 2 + seg_u
                seg = seg_u * u + seg_v * (2*v-u)*2 + m[i]
                
                tmp = seg[np.argmax(pred_density_mult(seg, pi, m, inv_sigma, coef))]   
                res = solve_ivp(fun=lambda t, y : pred_grad_with_t(t, y, pi, m, inv_sigma, coef), t_span=[0,0.1], y0 = tmp)
                m_tmp = res.y.T[1]
                
                sol = root(fun=pred_grad, x0=m_tmp, jac = pred_hess, args = (pi, m, inv_sigma, coef))
                tmp_x.append(sol.x)
            
    tmp_x = np.array(tmp_x)
    [dummy, I, J] = np.unique(np.round(10 * tmp_x), axis=0, return_index=True, return_inverse=True)
    tmp_x = tmp_x[I, :]
    for i in range(len(tmp_x)):
        sep = tmp_x[i]
        [f, g, H] = pred_density(sep, pi, m, inv_sigma, coef), pred_grad(sep, pi, m, inv_sigma, coef), pred_hess(sep, pi, m, inv_sigma, coef)
        [D, V] = np.linalg.eigh(H)
        ind = []
        if np.sum(D < 0) == 1:
            sep1 = sep + epsilon * V[np.where(D < 0)]
            sep2 = sep - epsilon * V[np.where(D < 0)]
            res1 = solve_ivp(fun=lambda t, y : pred_grad_with_t(t, y, pi, m, inv_sigma, coef), t_span=[0,1], y0 = sep1[0])
            temp1 = res1.y.T[-1]
            res2 = solve_ivp(fun=lambda t, y : pred_grad_with_t(t, y, pi, m, inv_sigma, coef), t_span=[0,1], y0 = sep2[0])
            temp2 = res2.y.T[-1]
            [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), m)),
                             np.argmin(euclidean_distances(temp1.reshape(1, -1), m))]
            [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), m)),
                             np.argmin(euclidean_distances(temp2.reshape(1, -1), m))]
            if ind1 != ind2:
                ts['x'].append(sep)
                ts['f'].append(f)
                ts['neighbor'].append([ind1, ind2])
                ts['purturb'].append([sep1, sep2])

    ts['x'] = np.array(ts['x'])
    ts['f'] = np.array(ts['f'])
    ts['neighbor'] = np.array(ts['neighbor'])
    ts['purturb'] = np.array(ts['purturb'])
    return ts

def hierarchicalLabel(K, m, ts, local_assignments):
    nOfLocals = m.shape[0]
    nOfTS = len(ts['f'])

    local_clusters_assignments = []
    f_sort = np.sort(ts['f'], 0)  # small --> large
    #print("f_sort:", f_sort)
    adjacent = np.zeros([nOfLocals, nOfLocals, nOfTS])
    a = []
    flag = 0
    for o in range(nOfTS):
        cur_f = f_sort[-o-1]  # % cutting level:large --> small  (small number of clusters --> large number of clusters)
        # %cur_f=f_sort[o];         % cutting level: small --> large (large number of clusters --> small number of clusters)

        tmp = np.nonzero(ts['f'] < cur_f)[0]
        if len(tmp) > 0: 
            for j in range(len(tmp)):
                adjacent[ts['neighbor'][tmp[j], 0], ts['neighbor'][tmp[j], 1], o] = 1
                adjacent[ts['neighbor'][tmp[j], 1], ts['neighbor'][tmp[j], 0], o] = 1
            for i in range(nOfLocals):
                for j in range(i):
                    if (adjacent[i, j, o] == 1):
                        adjacent[i, :, o] = np.logical_or(adjacent[i, :, o], adjacent[j, :, o])
                adjacent[i, i] = 1

        a = [a, cur_f]
        my_ts = {}
        my_ts['x'] = ts['x'][tmp, :]
        my_ts['f'] = ts['f'][tmp]
        my_ts['purturb'] = ts['purturb'][tmp, :]
        my_ts['neighbor'] = ts['neighbor'][tmp, :]
        my_ts['cuttingLevel'] = cur_f
        ind = np.nonzero(ts['f'] == cur_f)[0]
        my_ts['levelx'] = ts['x'][ind[0], :]
        tmp_ts = {} 
        tmp_ts[o] = my_ts

        assignment = cg.connected_components(adjacent[:, :, o])[1]
        #print("assignment:", assignment)
        #print("N_clusters:", np.max(assignment) + 1)
        if np.max(assignment) == K - 1:
            print('We can find the number of K clusters')
            out_ts = tmp_ts[o]
            local_ass = assignment
            cluster_labels = local_ass[local_assignments].T
            flag = 1
            return cluster_labels

        local_clusters_assignments.append(assignment)

    if flag == 0:
        print(
            'Cannot find cluster assignments with K number of clusters, instead that we find cluster assignments the with the nearest number of clusters to K !');
        ind = np.argmin(np.abs(np.max(local_clusters_assignments, 1)-K))
        local_clusters_assignments = local_clusters_assignments[ind]
        local_ass = local_clusters_assignments
        cluster_labels = local_ass[local_assignments]
        return cluster_labels