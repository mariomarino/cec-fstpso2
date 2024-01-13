import copy
import os
import sys

sys.path.insert(0, "./")
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
from benchmarks.soo.functions import functions
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def plot_boxplot(data: pd.DataFrame, dir_name: str, filename: str, function_name: str, x_name="x", y_name="y",
                 xlabel="x", ylabel="y", title="", order=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    sns.despine(ax=ax, offset=5, trim=False)
    # plt.style.use("seaborn-v0_8-colorblind")
    sns.boxplot(data=data, x=x_name, y=y_name, order=order, log_scale=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', labelsize=10, labelrotation=30)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.close()


def plotConvergence(gbests, dir_name, filename, labels, title="", labels_legend=None):
    plt.clf()
    # plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.title(title, fontsize=14)
    cmap = (plt.get_cmap("colorblind")).colors
    print(f"num colors: {len(cmap)}, num gbests: {len(gbests)}")
    for gb in range(len(gbests)):
        fevs = list(range(len(gbests[gb])))
        print(f"gb: {gb}")
        print(len(fevs), len(gbests[gb]))
        if labels[gb] == "stufstpso":
            plt.plot(fevs, gbests[gb], label=labels[gb], color="black")
        else:
            plt.plot(fevs, gbests[gb], label=labels[gb], color=cmap[gb])
    plt.xlabel("Fitness evaluation", fontsize=12)
    plt.ylabel("Best fitness", fontsize=12, c='black')
    plt.yscale("log")
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(fontsize=10, labels=labels_legend)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    print(f"{filename} :  created")
    plt.show()
    plt.clf()


def main(D=30, fitness="Rastrigin", budget_str="4B", budget=30 * 1e4, dir_results_base="results_CEC/remedies",
         runs=None,
         remedy_names=None, dir_baseline="results_CEC", remedy_name_label_map=None, plot_basins=False):
    if runs is None:
        runs = range(30)

    f_fitness = functions[fitness]

    sns_palette = sns.color_palette("colorblind", n_colors=12, as_cmap=True) + ["gray", "brown"]
    print(type(sns_palette))
    sns.set_palette(sns_palette)

    # register  colorblind palette
    colorblind = ListedColormap(sns_palette)
    matplotlib.colormaps.register(cmap=colorblind, name="colorblind")
    plt.set_cmap("colorblind")

    cmap = plt.get_cmap("colorblind")
    print("cmap")
    print(cmap.colors)

    dir_FSTPSO = f"{dir_baseline}/{fitness}/{budget_str}/gbest_FSTPSO"
    dir_RINGPSO = f"{dir_baseline}/{fitness}/{budget_str}/gbest_RINGPSO"

    labels_to_plot = ["RINGPSO", "FSTPSO"] + remedy_names

    gbest_FSTPSO = []
    gbest_RINGPSO = []

    with open(f"{dir_baseline}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{runs[0]}R_population.pickle",
              "rb") as f:
        initial_population = pickle.load(f)
    numparticles = len(initial_population)

    for R in runs:
        with open(f"{dir_FSTPSO}/{fitness}_{D}D_{R}R_gbest_FSTPSO.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_FSTPSO.append(np.repeat(tmp, numparticles))
        with open(f"{dir_RINGPSO}/{fitness}_{D}D_{R}R_gbest_RINGPSO.txt", "r") as f:
            tmp = [float(i) for i in f.read().splitlines()]
            gbest_RINGPSO.append(np.repeat(tmp, numparticles))

    gbest_remedies = []
    counter_exploration_remedies = []
    convergence_to_boxplot = []
    for remedy_name in remedy_names:
        gbest_remedy = []
        counter_exploration_remedy = []
        for R in runs:
            with open(f"{dir_baseline}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{R}R_population.pickle",
                      "rb") as f:
                initial_population = pickle.load(f)
            gbest_0 = np.min([f_fitness(s) for s in initial_population])

            rp = f"results_CEC/{fitness}/{budget_str}/gbest_{remedy_name}/{fitness}_{D}D_{R}R_gbest_{remedy_name}.txt"
            with open(rp, "r") as f:
                tmp = [float(i) for i in f.read().splitlines()]
                # tmp.insert(0, gbest_0)
                gbest_remedy.append(np.repeat(tmp, numparticles))
                convergence_to_boxplot.append({'method': remedy_name_label_map[remedy_name], 'fitness': tmp[-1]})
        gbest_remedies.append(gbest_remedy)

    obj_gbest_FSTPSO = {}
    obj_gbest_RINGPSO = {}
    for R in runs:
        obj_gbest_FSTPSO[R] = gbest_FSTPSO[R]
        obj_gbest_RINGPSO[R] = gbest_RINGPSO[R]
        convergence_to_boxplot.append({'method': remedy_name_label_map['ringpso'], 'fitness': gbest_RINGPSO[R][-1]})
        convergence_to_boxplot.append({'method': remedy_name_label_map['fstpso'], 'fitness': gbest_FSTPSO[R][-1]})

    obj_gbest_remedies = []
    for remedy_id in range(len(remedy_names)):
        obj_gbest_remedy = {}
        for R in runs:
            obj_gbest_remedy[R] = gbest_remedies[remedy_id][R]
        obj_gbest_remedies.append(obj_gbest_remedy)

    df_FSTPSO = pd.DataFrame(obj_gbest_FSTPSO)
    df_RINGPSO = pd.DataFrame(obj_gbest_RINGPSO)

    gbest_FSTPSO_fev = df_FSTPSO.median(axis=1).values.tolist()
    gbest_RINGPSO_fev = df_RINGPSO.median(axis=1).values.tolist()

    df_remedies = []
    gbest_remedies_fev = []
    for obj_gbest_remedy in obj_gbest_remedies:
        df = pd.DataFrame(obj_gbest_remedy)
        df_remedies.append(df)
        gbest_remedies_fev.append(df.median(axis=1).values.tolist())

    gbest_to_plot = [gbest_RINGPSO_fev, gbest_FSTPSO_fev] + gbest_remedies_fev
    for fevs in gbest_to_plot:
        print(f"+0: {fevs[0]}")
        print(f"len: {len(fevs)}")
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/convergence/png").mkdir(parents=True,
                                                                                               exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/convergence/pdf").mkdir(parents=True,
                                                                                               exist_ok=True)
    plotConvergence(gbest_to_plot,
                    dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/convergence/",
                    filename=f"{fitness}_{D}D_convergence",
                    labels=labels_to_plot,
                    title=fitness,
                    labels_legend=remedy_name_label_map.values())

    df_convergence = pd.DataFrame(convergence_to_boxplot)
    print("df convergence")
    print(df_convergence.head(5))
    plot_boxplot(
        data=df_convergence,
        dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/convergence",
        filename=f"{fitness}_{D}D_convergence_boxplot",
        function_name=fitness,
        x_name="method",
        y_name="fitness",
        xlabel="Method",
        ylabel="Fitness",
        title=fitness,
        order=list(remedy_name_label_map.values())
    )


def plotVelocities(VelocitiesOverTimeRead, dir_name="", filename=""):
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
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Velocity magnitude", fontsize=12, c='black')
    plt.tick_params(axis='both', labelsize=10)

    ax2 = plt.twinx()
    ax2.plot(xs, pointsAboveAvg, color='green')
    ax2.set_ylabel('Number of exploitative particles', fontsize=12)
    ax2.yaxis.label.set_color('green')
    ax2.tick_params(axis='y', colors='green', labelsize=10)
    plt.plot(pointsAboveAvg, c='green')
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png")
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf")
    print(f"{filename} :  created")
    plt.clf()
    plt.show()


def mainVelocities(D=30, fitness="Rastrigin", budget_str="4B", budget=30 * 1e4, dir_results_base="results/remedies",
                   runs=None,
                   remedy_names=None, dir_baseline="results/vanilla_50P", remedy_name_label_map=None):
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/png").mkdir(parents=True,
                                                                                              exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/pdf").mkdir(parents=True,
                                                                                              exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/png").mkdir(parents=True,
                                                                                              exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/pdf").mkdir(parents=True,
                                                                                              exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/png").mkdir(parents=True,
                                                                                              exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities/pdf").mkdir(parents=True,
                                                                                              exist_ok=True)

    dir_baseline_velocities = f"{dir_baseline}/{fitness}/{budget_str}/Velocities"

    # baseline_velocities = []
    with open(f"{dir_baseline_velocities}/{fitness}_{D}D_0R_velocities_PSO.pickle", "rb") as f:
        vel = pickle.load(f)
        print("plot velocity pso")
        plotVelocities(
            vel,
            f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities",
            f"{fitness}_{D}D_0R_velocity_pso"
        )
        # baseline_velocities.append(vel)
    with open(f"{dir_baseline_velocities}/{fitness}_{D}D_0R_velocities_RINGPSO.pickle", "rb") as f:
        vel = pickle.load(f)
        print("plot velocity ringpso")
        plotVelocities(
            vel,
            f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities",
            f"{fitness}_{D}D_0R_velocity_ringpso"
        )
        # baseline_velocities.append(vel)
    with open(f"{dir_baseline_velocities}/{fitness}_{D}D_0R_velocities_FSTPSO.pickle", "rb") as f:
        vel = pickle.load(f)
        print("plot velocity fstpso")
        plotVelocities(
            vel,
            f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities",
            f"{fitness}_{D}D_0R_velocity_fstpso"
        )
        # baseline_velocities.append(vel)

    labels_to_plot = ["PSO", "RINGPSO", "FSTPSO"] + remedy_names

    for remedy_name in remedy_names:
        dir_velocities = f"results/{remedy_name}/{fitness}/{budget_str}/Velocities"
        path = f"{dir_velocities}/{fitness}_{D}D_0R_velocities_{remedy_name}.pickle"
        with open(path, "rb") as f:
            vel = pickle.load(f)
            print(f"plot velocity {remedy_name}")
            plotVelocities(
                vel,
                f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/velocities",
                f"{fitness}_{D}D_0R_velocity_{remedy_name}"
            )
            # baseline_velocities.append(vel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=int)
    parser.add_argument("--dimensions", '-D', dest='D', type=int, default=2)
    parser.add_argument("--budget", '-B', dest='budget_str', type=str, default="4B")
    args = parser.parse_args()
    # pso_ring_dilation
    fitness = f"CEC17-F{args.fitness}"
    remedy_names = ["pso_ring_dilation_5", "pso_ring_dilation_6", "stufstpso"]
    remedy_name_label_map = {
        "ringpso": "RING-PSO",
        "fstpso": "FST-PSO",
        "pso_ring_dilation_5": "RING-PSO df_5",
        "pso_ring_dilation_6": "RING-PSO df_6",
        "stufstpso": "STU-PSO",
    }

    dir_results_base = "results_CEC/remedies_df"
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{args.budget_str}").mkdir(parents=True, exist_ok=True)

    main(D=args.D, fitness=fitness, budget_str=args.budget_str, budget=args.D * 1e4,
         dir_results_base=dir_results_base, remedy_names=remedy_names, remedy_name_label_map=remedy_name_label_map,
         plot_basins=True)
    """mainVelocities(D=args.D, fitness=args.fitness, budget_str=args.budget_str, budget=args.D * 1e4,
                   dir_results_base=dir_results_base, remedy_names=remedy_names,
                   remedy_name_label_map=remedy_name_label_map)"""
