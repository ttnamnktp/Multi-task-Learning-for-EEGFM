import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from experiments.toy.problem import Toy


# plotting utils
def plot_2d_pareto(trajectories: dict, scale):
    """Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""
    fig, ax = plt.subplots(figsize=(5, 5))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(res["traj"])))

    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).double()

    Ys = F.batch_forward(Xs)
    ax.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#72727A",
        #label="Pareto Front",
    )  # Pareto front

    for i, tt in enumerate(losses):
        ax.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            #label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        ax.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9)

    sns.despine()
    #ax.set_xlabel(r"$\ell_1$", size=30)
    #ax.set_ylabel(r"$\ell_2$", size=30)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    legend = ax.legend(
        loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
    )
    return ax, fig, legend

def compute_F_curve(J, n_points=200):

    g1 = J[:,0]
    g2 = J[:,1]

    alpha = torch.linspace(0,1,n_points)

    vals = []

    for a in alpha:
        z = torch.tensor([a,1-a])

        grad = g1*z[0] + g2*z[1]

        F = 0.5 * torch.sum(grad**2)

        vals.append(F.item())

    vals = torch.tensor(vals)

    # minimizer
    d = g1-g2

    denom = torch.sum(d*d)

    if denom > 1e-12:
        alpha_star = -torch.dot(g2,d)/denom
    else:
        alpha_star = 0.5

    alpha_star = torch.clamp(alpha_star,0,1)


    return (
        alpha.numpy(),
        vals.numpy(),
        alpha_star.item()
    )