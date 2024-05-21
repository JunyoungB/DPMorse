import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score as ars

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from GMM_hard_rate_final import GaussianMixtureHard
from utils import *


def main(k0, k, miter, eps, delta, deltai, init):
    inputs, outputs = make_moons(n_samples=10000, noise=0.05)

    X_sc = MinMaxScaler((-1,1)).fit_transform(inputs)
    n, D = X_sc.shape

    model = GaussianMixtureHard(K=k0, max_iter=miter, eps=eps, delta=delta, deltai=deltai, init_param=init)
    print("ARI score of DPMoG-hard : %.3f"%(ars(model.fit_predict(X_sc), outputs)))

    pi = model.pi[model.pi>1e-2]
    m = model.mu[model.pi>1e-2]
    sigma = model.sigma[model.pi>1e-2]

    inv_sigma = np.linalg.inv(sigma)
    det_s = np.linalg.det(sigma)
    coef = det_s*(2*np.pi)**D
    coef = 1/(coef**0.5)

    x = np.arange(X_sc[:,0].min()-0.1, X_sc[:,0].max()+0.1, 0.01)
    y = np.arange(X_sc[:,1].min()-0.1, X_sc[:,1].max()+0.1, 0.01)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()]).T

    Z = pred_density_mult(positions, pi, m, inv_sigma, coef).reshape(len(y), len(x))

    plt.figure(figsize=(9,6))
    colors = ['r','g','b']
    j=0
    for i in np.unique(outputs):
        plt.scatter(X_sc.T[0][outputs==i], X_sc.T[1][outputs==i], color=colors[j], marker = 'o', alpha=0.08)
        j+=1
    plt.scatter(m.T[0], m.T[1], s=50, color='black')
    plt.contour(X, Y, Z)
    plt.xticks(ticks = [-1,-0.5,0,0.5,1],fontsize = 15)
    plt.yticks(ticks = [-1,-0.5,0,0.5,1],fontsize = 15)
    plt.savefig('before_connect.png', dpi=300, bbox_inches='tight')

    ts = findTPs(pi, m, inv_sigma, coef)

    plt.figure(figsize=(9,6))
    colors = ['r','g','b']
    j=0
    for i in np.unique(outputs):
        plt.scatter(X_sc.T[0][outputs==i], X_sc.T[1][outputs==i], color=colors[j], marker = 'o', alpha=0.05)
        j+=1
    plt.scatter(m.T[0], m.T[1], color='black')
    plt.scatter(ts['x'].T[0], ts['x'].T[1], s=70, marker='x', color='k')
    for i, txt in enumerate(ts['f']):
        plt.annotate(str(np.round(txt,3))+', ('+str(ts['neighbor'][i][0])+', '+str(ts['neighbor'][i][1])+')', ts['x'][i]+np.array([-0.18, -0.11]), fontsize=15)
    for i, _ in enumerate(m):
        plt.annotate(i, m[i]+np.array([0.01,0.01]), fontsize=15)
    plt.xticks(ticks = [-1,-0.5,0,0.5,1],fontsize = 15)
    plt.yticks(ticks = [-1,-0.5,0,0.5,1],fontsize = 15)
    plt.savefig('after_connect.png', dpi=300, bbox_inches='tight')

    scores = pred_density_score_mult(X_sc, pi, m, inv_sigma, coef)
    local_assignments = np.argmax(scores, axis=1)

    print("ARI score of DPMoG-hard-morse : %.3f"%(ars(hierarchicalLabel(k, m, ts, local_assignments), outputs)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Main method", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--subclusters', type=int, default=8, help='number of sub-clusters')
    parser.add_argument('--clusters', type=int, default=2, help='number of final clusters')
    parser.add_argument('--iter', type=int, default=10, help='number of iteration for DPMoG-hard')
    parser.add_argument('--eps', type=float, default=10, help='privacy budget')
    parser.add_argument('--delta', type=float, default=1e-4, help='privacy budget')
    parser.add_argument('--deltai', type=float, default=1e-6, help='privacy budget')
    parser.add_argument('--init', type=str, default='smart', choices=['smart', 'random'], help='method for center initialization')
    args = parser.parse_args()
    main(args.subclusters, args.clusters, args.iter, args.eps, args.delta, args.deltai, args.init)