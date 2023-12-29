import os
import pathlib
import pickle
import argparse

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def plotConvergence(gbests, f_name, title="", save=False):
    fevs = list(range(len(gbests[0])))
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    plt.plot(fevs, gbests[0], label="standard")
    plt.plot(fevs, gbests[1], label="perturbation")
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
    print("data")
    print(data)
    print("arr2d")
    print(arr2d)
    # exit()
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    # ax.imshow(data, cmap="colorblind", origin="lower", vmin=0)
    # ax.grid(which="minor")

    plt.ylabel("Iteration")
    plt.xlabel("Particle")

    hm = sns.heatmap(data)
    plt.gca().invert_yaxis()
    plt.xticks(list(range(len(data[0]))), [str(l) for l in range(1, len(data[0]) + 1)])
    colorbar = hm.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / 5 + r * i / 5 for i in range(5)])
    colorbar.set_ticklabels(["SE", "SR", "DE", "FE", "EX"])
    # plt.legend()
    # plt.scatter(x, y, s=2, c=z, cmap="colorblind", marker="1", linewidths=1)
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.clf()
    plt.close()


def plotBasins(basins_iters_nostall, basins_iters_stall, title="", f_name="", save=True):
    n_basins_0_nostall = len(np.unique(basins_iters_nostall[0]))
    n_iters = int(0.6*len(basins_iters_nostall))
    n_new_basins_iter_nostall = [n_basins_0_nostall]
    for iteration in range(1, n_iters):  # len(basins_iters_nostall)
        flat_list = [basin for sublist in basins_iters_nostall[0:iteration] for basin in sublist]
        # flat_list = [basin for basin in basins_iters_nostall[iteration]]
        n_new_basins_iter_nostall.append(len(np.unique(flat_list)))

    n_basins_0_stall = len(np.unique(basins_iters_stall[0]))
    n_new_basins_iter_stall = [n_basins_0_stall]
    for iteration in range(1, n_iters):
        flat_list = [basin for sublist in basins_iters_stall[0:iteration] for basin in sublist]
        # flat_list = [basin for basin in basins_iters_stall[iteration]]
        n_new_basins_iter_stall.append(len(np.unique(flat_list)))

    iters = range(1, n_iters+1)  # len(basins_iters_nostall)+1
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title)
    plt.plot(iters, n_new_basins_iter_nostall, label="standard")
    plt.plot(iters, n_new_basins_iter_stall, label="perturbation")
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

    dir_nostall = f"{dir_results_base}/{fitness}/{budget_str}/gbest_nostall"
    dir_stall = f"{dir_results_base}/{fitness}/{budget_str}/gbest_stall"
    gbest_nostall = []
    gbest_stall = []
    counter_exploration_stall = []
    counter_exploration_nostall = []

    with open(f"{dir_results_base}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{runs[0]}R_population.pickle",
              "rb") as f:
        population_0 = pickle.load(f)
    numparticles = len(population_0)
    for R in runs:
        with open(f"{dir_nostall}/{fitness}_{D}D_{R}R_gbest_nostall.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_nostall.append(np.repeat(tmp, numparticles))
        with open(f"{dir_stall}/{fitness}_{D}D_{R}R_gbest_stall.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_stall.append(np.repeat(tmp, numparticles))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_nostall.pickle",
                "rb") as f:
            counter_exploration_nostall.append(pickle.load(f))
        with open(
                f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_stall.pickle",
                "rb") as f:
            counter_exploration_stall.append(pickle.load(f))

    obj_gbest_nostall = {}
    for R in runs:
        obj_gbest_nostall[R] = gbest_nostall[R]
    obj_gbest_stall = {}
    for R in runs:
        obj_gbest_stall[R] = gbest_stall[R]
    df_nostall = pd.DataFrame(obj_gbest_nostall)
    df_stall = pd.DataFrame(obj_gbest_stall)

    # df_nostall.median(axis=1)
    # df_stall.median(axis=1)
    # gbest_nostall_fev = df_nostall[df_nostall.columns[0]].values.tolist()
    # gbest_stall_fev = df_stall[df_stall.columns[0]].values.tolist()
    gbest_nostall_fev = df_nostall.median(axis=1).values.tolist()
    gbest_stall_fev = df_stall.median(axis=1).values.tolist()

    plotConvergence([gbest_nostall_fev, gbest_stall_fev],
                    f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_convergence",
                    title=fitness, save=True)
    exit()
    counter_exploration_nostall = np.array(counter_exploration_nostall)
    counter_exploration_stall = np.array(counter_exploration_stall)

    cols = ["SE", "DE", "FE", "SR", "EX"]
    data_counter = []
    budget = 1

    for c in counter_exploration_nostall:
        print(f"c: {c}")
        data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': 'no stall'})
        data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': 'no stall'})
        data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': 'no stall'})
        data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': 'no stall'})
        data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': 'no stall'})

    for c in counter_exploration_stall:
        data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': 'stall'})
        data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': 'stall'})
        data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': 'stall'})
        data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': 'stall'})
        data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': 'stall'})

    """df_counter_nostall = pd.DataFrame(counter_exploration_nostall, columns=cols)
    df_counter_nostall["Method"] = ["No stall" for _ in counter_exploration_nostall]
    df_counter_stall = pd.DataFrame(counter_exploration_stall, columns=cols)
    df_counter_stall["Method"] = ["Stall" for _ in counter_exploration_stall]

    df_counter = pd.concat([df_counter_nostall, df_counter_stall])"""
    df_counter = pd.DataFrame(data_counter)
    print("here A")
    plotCounter(data=df_counter,
                f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_counters",
                save=True)
    print("here B")
    print(counter_exploration_stall)

    R = 0
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_nostall.pickle",
            "rb") as f:
        particles_counters_nostall = pickle.load(f)
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_stall.pickle",
            "rb") as f:
        particles_counters_stall = pickle.load(f)
    plotParticlesCounters(data=particles_counters_nostall,
                          f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_PCN",
                          save=True)
    plotParticlesCounters(data=particles_counters_stall,
                          f_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_PCS",
                          save=True)
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_nostall.pickle",
            "rb") as f:
        basins_iters_nostall = pickle.load(f)
    with open(
            f"./{dir_results_base}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_stall.pickle",
            "rb") as f:
        basins_iters_stall = pickle.load(f)
    plotBasins(basins_iters_nostall,
               basins_iters_stall,
               "N basins found",
               f"{dir_results_base}/plots/{fitness}/{budget_str}/{fitness}_{D}D_{R}R_new_basins",
               True)


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


def mainVelocities(D=30, fitness="Rastrigin", budget_str="4B"):
    dir_results_base = f"../tmp_analysis"
    dir_deltavelocity = f"{dir_results_base}/{fitness}/{budget_str}/DeltaVelocity"
    with open(f"{dir_deltavelocity}/{fitness}_{D}D_0R_DeltaVelocity.pickle", "rb") as f:
        DeltaVelocity = pickle.load(f)
    plotDeltaVelocity(DeltaVelocity, f"{dir_results_base}/plots/{fitness}_{D}D_0R_DeltaVelocity",
                      title="Delta velocity", save=True)


def mainVelocities2(D=30, fitness="Rastrigin", budget_str="4B", R=0, dir_results_base="./tmp_analysis"):
    with open(
            f"{dir_results_base}/{fitness}/{budget_str}/Velocities/{fitness}_{D}D_{R}R_velocities_stall.pickle",
            "rb") as f:
        velocities_iter_stall = pickle.load(f)
        plotVelocities(velocities_iter_stall, f"{dir_results_base}/plots/{fitness}_{D}D_{R}R_Velocities_stall",
                       save=True)
    with open(
            f"{dir_results_base}/{fitness}/{budget_str}/Velocities/{fitness}_{D}D_{R}R_velocities_nostall.pickle",
            "rb") as f:
        velocities_iter_nostall = pickle.load(f)
        print("here")

        plotVelocities(velocities_iter_nostall, f"{dir_results_base}/plots/{fitness}_{D}D_{R}R_Velocities_nostall",
                       save=True)


def mainFactors(D=30, fitness="Rastrigin", budget_str="4B"):
    dir_results_base = f"../tmp_analysis"
    dir_factors = f"{dir_results_base}/{fitness}/{budget_str}/Factors"
    # reference
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Social_r.pickle", "rb") as f:
        Social_r = pickle.load(f)
        plotDeltaVelocity(Social_r, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Social_r",
                          title="Social factor, reference", save=True)
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Cognitive_r.pickle", "rb") as f:
        Cognitive_r = pickle.load(f)
        plotDeltaVelocity(Cognitive_r, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Cognitive_r",
                          title="Cognitive factor, reference", save=True)
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Inertia_r.pickle", "rb") as f:
        Inertia_r = pickle.load(f)
        plotDeltaVelocity(Inertia_r, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Inertia_r",
                          title="Inertia factor, reference", save=True)

    # perturb
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Social_p.pickle", "rb") as f:
        Social_p = pickle.load(f)
        plotDeltaVelocity(Social_p, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Social_p",
                          title="Social factor, after perturb", save=True)
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Cognitive_p.pickle", "rb") as f:
        Cognitive_p = pickle.load(f)
        plotDeltaVelocity(Cognitive_p, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Cognitive_p",
                          title="Cognitive factor, after perturb", save=True)
    with open(f"{dir_factors}/{fitness}_{D}D_0R_Inertia_p.pickle", "rb") as f:
        Inertia_p = pickle.load(f)
        plotDeltaVelocity(Inertia_p, f"{dir_results_base}/plots/{fitness}_{D}D_0R_Inertia_p",
                          title="Inertia factor, after perturb", save=True)


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

    # main(args.D, args.fitness, args.budget_stWr)
    # mainFactors(args.D, args.fitness, args.budget_str)
    # mainVelocities(args.D, args.fitness, args.budget_str)
    main(D=args.D, fitness=args.fitness, budget_str=args.budget_str, budget=args.D * 1e4,
         dir_results_base="./results/vanilla")  # , runs=[0]
    # mainVelocities2(D=args.D, fitness=args.fitness, budget_str=args.budget_str, R=0, dir_results_base="./results")
