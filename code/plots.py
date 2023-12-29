import os
import pathlib
import pickle
import argparse

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def plotConvergence(gbests, f_name, labels, title="", save=False):
    fevs = list(range(len(gbests[0])))
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    """plt.plot(fevs, gbests[0], label=labels[0])
    plt.plot(fevs, gbests[1], label=labels[1])
    plt.plot(fevs, gbests[2], label=labels[2])"""
    for gb in range(len(gbests)):
        plt.plot(fevs, gbests[gb], label=labels[gb])
    plt.xlabel("Fitness evaluation", fontsize=18)
    plt.ylabel("Best fitness", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.show()
    plt.clf()


def plotCounter(data, f_name, save=False):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="type", y="value", hue="method")
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.clf()
    plt.close()


def plotParticlesCounters(data, f_name="", save=False):
    toscatter = []
    for i_iteration in range(len(data)):
        iteration = data[i_iteration]
        for i_p in range(len(iteration)):
            toscatter.append((i_p, i_iteration, iteration[i_p]))
    x, y, z = zip(*toscatter)
    arr2d = np.array(data)
    # exit()
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(14)
    # ax.imshow(data, cmap="colorblind", origin="lower", vmin=0)
    # ax.grid(which="minor")

    plt.ylabel("Iteration")
    plt.xlabel("Particle")


    """plt.gca().invert_yaxis()
    plt.xticks(list(range(len(data[0]))), [str(l) for l in range(1, len(data[0]) + 1)])
    colorbar = hm.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / 5 + r * i / 5 for i in range(5)])
    colorbar.set_ticklabels(["SE", "SR", "DE", "FE", "EX"])"""

    # 0 -> Successful Exploration   -> red
    # 1 -> Deceptive Exploration    -> yellow
    # 2 -> Failed Exploration       -> black
    # 3 -> Successful Rejection     -> blue
    # 4 -> Exploitation             -> gray

    cmap = colors.ListedColormap(["red", "yellow", "black", "blue", "lightgray"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    hm = sns.heatmap(data, ax=ax, cmap=cmap, norm=norm, cbar=True)
    plt.gca().invert_yaxis()
    plt.xticks(list(range(len(data[0]))), [str(l) for l in range(1, len(data[0]) + 1)])
    colorbar = hm.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / 5 + r * i / 5 for i in range(5)])
    colorbar.set_ticklabels([
                     "Successful Exploration",
                     "Deceptive Exploration",
                     "Failed Exploration",
                     "Successful Rejection",
                     "Exploitation"])
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.clf()
    plt.close()


def basins_to_count(basins):
    n_iters = len(basins)
    flat_list = set([])
    count_basins = []
    for iteration in range(n_iters - 1):
        flat_list = flat_list.union(basins[iteration])
        count_basins.append(len(flat_list))
    return count_basins


def plotBasins2(basins, labels=None, title="", f_name=""):
    iters = range(1, len(basins[0]) + 1)

    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    for b in range(len(basins)):
        plt.plot(iters, basins[b], label=labels[b])
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("N", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{f_name}.png")
    plt.savefig(f"{f_name}.pdf")
    print(f"{f_name} :  created")
    plt.clf()


def plotBasins(basins_iters_PSO, basins_iters_FSTPSO, basins_iters_RINGPSO, title="", f_name="", save=True):
    n_basins_0_PSO = len(np.unique(basins_iters_PSO[0]))
    n_iters = int(len(basins_iters_PSO))  #
    n_new_basins_iter_PSO = [n_basins_0_PSO]
    flat_list = set([])
    for iteration in range(n_iters - 1):
        flat_list = flat_list.union(set([basin for sublist in basins_iters_PSO[iteration] for basin in sublist]))
        n_new_basins_iter_PSO.append(len(flat_list))

    n_basins_0_FSTPSO = len(np.unique(basins_iters_FSTPSO[0]))
    n_new_basins_iter_FSTPSO = [n_basins_0_FSTPSO]
    flat_list = set([])
    print("RINGPSO SETS")
    for iteration in range(n_iters - 1):
        flat_list = flat_list.union(set([basin for sublist in basins_iters_FSTPSO[iteration] for basin in sublist]))
        n_new_basins_iter_FSTPSO.append(len(flat_list))

    n_basins_0_RINGPSO = len(np.unique(basins_iters_RINGPSO[0]))
    n_new_basins_iter_RINGPSO = [n_basins_0_RINGPSO]
    flat_list = set([])
    for iteration in range(n_iters - 1):
        flat_list = flat_list.union(set([basin for sublist in basins_iters_RINGPSO[iteration] for basin in sublist]))
        n_new_basins_iter_RINGPSO.append(len(flat_list))

    iters = range(1, n_iters + 1)
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    plt.plot(iters, n_new_basins_iter_PSO, label="PSO")
    plt.plot(iters, n_new_basins_iter_FSTPSO, label="FSTPSO")
    plt.plot(iters, n_new_basins_iter_RINGPSO, label="RINGPSO")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("N", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.show()
    plt.clf()


def main(D=30, fitness="Rastrigin", budget_str="4B", budget=30 * 1e4, dir_results_base=f"./results_3_005", runs=None):
    if runs is None:
        runs = range(30)

    # register  colorblind palette
    colorblind = ListedColormap(sns.color_palette("colorblind", 5))
    matplotlib.colormaps.register(cmap=colorblind, name="colorblind")

    dir_PSO = f"{dir_results_base}/{fitness}/{budget_str}/gbest_PSO"
    dir_FSTPSO = f"{dir_results_base}/{fitness}/{budget_str}/gbest_FSTPSO"
    dir_RINGPSO = f"{dir_results_base}/{fitness}/{budget_str}/gbest_RINGPSO"
    gbest_PSO = []
    gbest_FSTPSO = []
    gbest_RINGPSO = []
    counter_exploration_FSTPSO = []
    counter_exploration_PSO = []
    counter_exploration_RINGPSO = []

    with open(f"{dir_results_base}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{runs[0]}R_population.pickle",
              "rb") as f:
        population_0 = pickle.load(f)
    numparticles = len(population_0)
    for R in runs:
        with open(f"{dir_PSO}/{fitness}_{D}D_{R}R_gbest_PSO.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_PSO.append(np.repeat(tmp, numparticles))
        with open(f"{dir_FSTPSO}/{fitness}_{D}D_{R}R_gbest_FSTPSO.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_FSTPSO.append(np.repeat(tmp, numparticles))
        with open(f"{dir_RINGPSO}/{fitness}_{D}D_{R}R_gbest_RINGPSO.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_RINGPSO.append(np.repeat(tmp, numparticles))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_PSO.pickle",
                "rb") as f:
            counter_exploration_PSO.append(pickle.load(f))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_FSTPSO.pickle",
                "rb") as f:
            counter_exploration_FSTPSO.append(pickle.load(f))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_RINGPSO.pickle",
                "rb") as f:
            counter_exploration_RINGPSO.append(pickle.load(f))

    obj_gbest_PSO = {}
    for R in runs:
        obj_gbest_PSO[R] = gbest_PSO[R]
    obj_gbest_FSTPSO = {}
    for R in runs:
        obj_gbest_FSTPSO[R] = gbest_FSTPSO[R]
    obj_gbest_RINGPSO = {}
    for R in runs:
        obj_gbest_RINGPSO[R] = gbest_RINGPSO[R]

    df_PSO = pd.DataFrame(obj_gbest_PSO)
    df_FSTPSO = pd.DataFrame(obj_gbest_FSTPSO)
    df_RINGPSO = pd.DataFrame(obj_gbest_RINGPSO)

    gbest_PSO_fev = df_PSO.median(axis=1).values.tolist()
    gbest_FSTPSO_fev = df_FSTPSO.median(axis=1).values.tolist()
    gbest_RINGPSO_fev = df_RINGPSO.median(axis=1).values.tolist()

    plotConvergence([gbest_PSO_fev, gbest_RINGPSO_fev, gbest_FSTPSO_fev],
                    f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_convergence",
                    ["PSO", "RINGPSO", "FSTPSO"],
                    title=fitness, save=True)
    counter_exploration_PSO = np.array(counter_exploration_PSO)
    counter_exploration_FSTPSO = np.array(counter_exploration_FSTPSO)
    counter_exploration_RINGPSO = np.array(counter_exploration_RINGPSO)

    cols = ["SE", "DE", "FE", "SR", "EX"]
    data_counter = []
    budget = 1

    for c in counter_exploration_PSO:
        data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': 'PSO'})
        data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': 'PSO'})
        data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': 'PSO'})
        data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': 'PSO'})
        data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': 'PSO'})

    for c in counter_exploration_FSTPSO:
        data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': 'FSTPSO'})
        data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': 'FSTPSO'})
        data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': 'FSTPSO'})
        data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': 'FSTPSO'})
        data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': 'FSTPSO'})

    for c in counter_exploration_RINGPSO:
        data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': 'RINGPSO'})
        data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': 'RINGPSO'})
        data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': 'RINGPSO'})
        data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': 'RINGPSO'})
        data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': 'RINGPSO'})

    df_counter = pd.DataFrame(data_counter)

    plotCounter(data=df_counter,
                f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_counters",
                save=True)

    R = 0
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_PSO.pickle",
            "rb") as f:
        particles_counters_PSO = pickle.load(f)
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_FSTPSO.pickle",
            "rb") as f:
        particles_counters_FSTPSO = pickle.load(f)
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_RINGPSO.pickle",
            "rb") as f:
        particles_counters_RINGPSO = pickle.load(f)

    plotParticlesCounters(data=particles_counters_PSO,
                          f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_PC_PSO",
                          save=True)
    plotParticlesCounters(data=particles_counters_FSTPSO,
                          f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_PC_FSTPSO",
                          save=True)
    plotParticlesCounters(data=particles_counters_RINGPSO,
                          f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_PC_RINGPSO",
                          save=True)

    basins_PSO = []
    basins_FSTPSO = []
    basins_RINGPSO = []

    for R in runs:
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_PSO.pickle",
                "rb") as f:
            basins_PSO.append(basins_to_count(pickle.load(f)))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_FSTPSO.pickle",
                "rb") as f:
            basins_FSTPSO.append(basins_to_count(pickle.load(f)))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_RINGPSO.pickle",
                "rb") as f:
            basins_RINGPSO.append(basins_to_count(pickle.load(f)))

    obj_basins_PSO = {}
    for R in runs:
        obj_basins_PSO[R] = basins_PSO[R]
    obj_basins_FSTPSO = {}
    for R in runs:
        obj_basins_FSTPSO[R] = basins_FSTPSO[R]
    obj_basins_RINGPSO = {}
    for R in runs:
        obj_basins_RINGPSO[R] = basins_RINGPSO[R]

    df_basins_PSO = pd.DataFrame(obj_basins_PSO)
    df_basins_FSTPSO = pd.DataFrame(obj_basins_FSTPSO)
    df_basins_RINGPSO = pd.DataFrame(obj_basins_RINGPSO)

    basins_PSO_median = df_basins_PSO.median(axis=1).values.tolist()
    basins_FSTPSO_median = df_basins_FSTPSO.median(axis=1).values.tolist()
    basins_RINGPSO_median = df_basins_RINGPSO.median(axis=1).values.tolist()
    print("lens")
    print(len(basins_PSO_median), len(basins_RINGPSO_median), len(basins_FSTPSO_median))
    plotBasins2(
        [basins_PSO_median, basins_RINGPSO_median, basins_FSTPSO_median],
        ["PSO", "RINGPSO", "FSTPSO"],
        "Cumulative N of explored basins",
        f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_new_basins"
    )


def plotDeltaVelocity(data, f_name, title="", save=False):
    iterations = list(range(len(data)))
    print(f"n_iters: {len(iterations)}")
    toscatter = []
    plt.clf()
    for i in range(len(iterations)):
        for j in data[i]:
            toscatter.append((i, j))
    print(f"n scatter: {len(toscatter)}")

    """avgs = np.array([np.average(v) for v in DeltaVelocity])
    stds = np.array([np.std(v) for v in DeltaVelocity])"""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    """plt.plot(iterations, avgs, color="black")
    plt.fill_between(iterations, avgs - stds, avgs + stds, color="C0", alpha=0.5)"""
    x, y = zip(*toscatter)
    print(f"x: {len(x)}")
    print(f"y: {len(y)}")
    plt.scatter(x, y, marker=".", s=1)
    plt.xlabel("Iterations", fontsize=18)
    plt.ylabel("|Vp - V|", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)
    # plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.clf()
    plt.show()


def plotVelocities(VelocitiesOverTimeRead, f_name, save=False):
    velocitiesOverTime = []
    previousVelocities = []
    pointsAboveAvg = []
    plt.clf()
    k = 0
    for i in VelocitiesOverTimeRead:
        k += 1
        previousVelocities = previousVelocities + i
        mean_vel = np.mean(previousVelocities)
        velocitiesOverTime.append(mean_vel)
        pointsAboveAvg.append(len([h for h in i if h < mean_vel]))

    xs = list(range(0, len(VelocitiesOverTimeRead)))
    plt.figure(figsize=(10, 6))
    for xe, ye in zip(xs, VelocitiesOverTimeRead):
        plt.scatter([xe] * len(ye), ye, c='royalblue', s=1)

    plt.tick_params(axis='y', colors='black')
    plt.plot(velocitiesOverTime, c='red')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Velocity magnitude", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)

    ax2 = plt.twinx()
    ax2.plot(xs, pointsAboveAvg, color='green')
    ax2.set_ylabel('Number of exploitative particles', fontsize=18)
    ax2.yaxis.label.set_color('green')
    ax2.tick_params(axis='y', colors='green', labelsize=14)
    plt.plot(pointsAboveAvg, c='green')
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.clf()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=str, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int, default=2)
    parser.add_argument("--budget", '-B', dest='budget_str', type=str, default="4B")
    args = parser.parse_args()

    main(D=args.D, fitness=args.fitness, budget_str=args.budget_str, budget=args.D * 1e4,
         dir_results_base="./results/vanilla_50P")  # , runs=[0]
