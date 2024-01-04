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
    sns.boxplot(data=data, x=x_name, y=y_name, order=order)
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
        plt.plot(fevs, gbests[gb], label=labels[gb], color=cmap[gb])
    plt.xlabel("Fitness evaluation", fontsize=12)
    plt.ylabel("Best fitness", fontsize=12, c='black')
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(fontsize=10, labels=labels_legend)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    print(f"{filename} :  created")
    plt.show()
    plt.clf()


def plotCounter(data, dir_name, filename, title="", xlabel="x", ylabel="y"):
    plt.figure(figsize=(10, 6))
    # plt.style.use("seaborn-v0_8-colorblind")
    sns.boxplot(data=data, x="type", y="value", hue="method")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    print(f"{filename} :  created")
    plt.clf()
    plt.close()


def plotParticlesCounters(data, dir_name="", filename="", title=""):
    toscatter = []
    for i_iteration in range(len(data)):
        iteration = data[i_iteration]
        for i_p in range(len(iteration)):
            toscatter.append((i_p, i_iteration, iteration[i_p]))
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(14)

    # 0 -> Successful Exploration   -> red
    # 1 -> Deceptive Exploration    -> yellow
    # 2 -> Failed Exploration       -> black
    # 3 -> Successful Rejection     -> blue
    # 4 -> Exploitation             -> gray
    # plt.style.use("seaborn-v0_8-colorblind")
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
    plt.ylabel("Iteration", fontsize=12)
    plt.xlabel("Particle", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    plt.clf()
    plt.close()


def basins_to_count_cum(basins):
    n_iters = len(basins)
    flat_list = set([])
    count_basins = []
    for iteration in range(n_iters):
        flat_list = flat_list.union(basins[iteration])
        count_basins.append(len(flat_list))
    return count_basins


def basins_to_count(basins):
    n_iters = len(basins)
    count_basins = []
    for iteration in range(n_iters):
        count_basins.append(len(basins[iteration]))
    return count_basins


def plotBasins(basins, labels=None, title="", dir_name="", filename="", labels_legend=None):
    iters = range(1, len(basins[0]) + 1)

    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    # plt.style.use("seaborn-v0_8-colorblind")
    plt.title(title, fontsize="14")
    cmap = (plt.get_cmap("colorblind")).colors
    for b in range(len(basins)):
        plt.plot(iters, basins[b], label=labels[b], color=cmap[b])
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("N", fontsize=12, c='black')
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(fontsize=10, labels=labels_legend)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/png/{filename}.png", dpi=600)
    plt.savefig(f"{dir_name}/pdf/{filename}.pdf", dpi=600)
    print(f"{filename} :  created")
    plt.clf()


def main(D=30, fitness="Rastrigin", budget_str="4B", budget=30 * 1e4, dir_results_base="results/remedies", runs=None,
         remedy_names=None, dir_baseline="results/vanilla_50P", remedy_name_label_map=None):
    if runs is None:
        runs = range(30)

    f_fitness = functions[fitness]

    sns_palette = sns.color_palette("colorblind", n_colors=12, as_cmap=True) + ["gray", "black"]
    print(type(sns_palette))
    sns.set_palette(sns_palette)

    # register  colorblind palette
    colorblind = ListedColormap(sns_palette)
    matplotlib.colormaps.register(cmap=colorblind, name="colorblind")
    plt.set_cmap("colorblind")

    cmap = plt.get_cmap("colorblind")
    print("cmap")
    print(cmap.colors)

    dir_PSO = f"{dir_baseline}/{fitness}/{budget_str}/gbest_PSO"
    dir_FSTPSO = f"{dir_baseline}/{fitness}/{budget_str}/gbest_FSTPSO"
    dir_RINGPSO = f"{dir_baseline}/{fitness}/{budget_str}/gbest_RINGPSO"

    labels_to_plot = ["PSO", "RINGPSO", "FSTPSO"] + remedy_names

    gbest_PSO = []
    gbest_FSTPSO = []
    gbest_RINGPSO = []
    counter_exploration_FSTPSO = []
    counter_exploration_PSO = []
    counter_exploration_RINGPSO = []

    with open(f"{dir_baseline}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{runs[0]}R_population.pickle",
              "rb") as f:
        initial_population = pickle.load(f)
    numparticles = len(initial_population)

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
                f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_PSO.pickle",
                "rb") as f:
            counter_exploration_PSO.append(pickle.load(f))
        with open(
                f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_FSTPSO.pickle",
                "rb") as f:
            counter_exploration_FSTPSO.append(pickle.load(f))
        with open(
                f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_RINGPSO.pickle",
                "rb") as f:
            counter_exploration_RINGPSO.append(pickle.load(f))

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

            rp = f"results/{remedy_name}/{fitness}/{budget_str}/gbest_{remedy_name}/{fitness}_{D}D_{R}R_gbest_{remedy_name}.txt"
            with open(rp, "r") as f:
                tmp = [float(i) for i in f.read().splitlines()]
                # tmp.insert(0, gbest_0)
                gbest_remedy.append(np.repeat(tmp, numparticles))
                convergence_to_boxplot.append({'method': remedy_name_label_map[remedy_name], 'fitness': tmp[-1]})
            rp = f"results/{remedy_name}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_counter_exploration_{remedy_name}.pickle"
            with open(
                    rp,
                    "rb") as f:
                counter_exploration_remedy.append(pickle.load(f))
        gbest_remedies.append(gbest_remedy)
        counter_exploration_remedies.append(counter_exploration_remedy)

    obj_gbest_PSO = {}
    obj_gbest_FSTPSO = {}
    obj_gbest_RINGPSO = {}
    for R in runs:
        obj_gbest_PSO[R] = gbest_PSO[R]
        obj_gbest_FSTPSO[R] = gbest_FSTPSO[R]
        obj_gbest_RINGPSO[R] = gbest_RINGPSO[R]
        convergence_to_boxplot.append({'method': remedy_name_label_map['pso'], 'fitness': gbest_PSO[R][-1]})
        convergence_to_boxplot.append({'method': remedy_name_label_map['ringpso'], 'fitness': gbest_RINGPSO[R][-1]})
        convergence_to_boxplot.append({'method': remedy_name_label_map['fstpso'], 'fitness': gbest_FSTPSO[R][-1]})

    obj_gbest_remedies = []
    for remedy_id in range(len(remedy_names)):
        obj_gbest_remedy = {}
        for R in runs:
            obj_gbest_remedy[R] = gbest_remedies[remedy_id][R]
        obj_gbest_remedies.append(obj_gbest_remedy)

    df_PSO = pd.DataFrame(obj_gbest_PSO)
    df_FSTPSO = pd.DataFrame(obj_gbest_FSTPSO)
    df_RINGPSO = pd.DataFrame(obj_gbest_RINGPSO)

    gbest_PSO_fev = df_PSO.median(axis=1).values.tolist()
    gbest_FSTPSO_fev = df_FSTPSO.median(axis=1).values.tolist()
    gbest_RINGPSO_fev = df_RINGPSO.median(axis=1).values.tolist()

    df_remedies = []
    gbest_remedies_fev = []
    for obj_gbest_remedy in obj_gbest_remedies:
        df = pd.DataFrame(obj_gbest_remedy)
        df_remedies.append(df)
        gbest_remedies_fev.append(df.median(axis=1).values.tolist())

    gbest_to_plot = [gbest_PSO_fev, gbest_RINGPSO_fev, gbest_FSTPSO_fev] + gbest_remedies_fev
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

    counter_exploration_PSO = np.array(counter_exploration_PSO)
    counter_exploration_FSTPSO = np.array(counter_exploration_FSTPSO)
    counter_exploration_RINGPSO = np.array(counter_exploration_RINGPSO)

    for id_remedy in range(len(counter_exploration_remedies)):
        counter_exploration_remedies[id_remedy] = np.array(counter_exploration_remedies[id_remedy])

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

    for remedy_id in range(len(remedy_names)):
        remedy_name = remedy_names[remedy_id]
        counter_exploration_remedy = counter_exploration_remedies[remedy_id]
        for c in counter_exploration_remedy:
            data_counter.append({'value': c[0] / budget, 'type': cols[0], 'method': remedy_name})
            data_counter.append({'value': c[1] / budget, 'type': cols[1], 'method': remedy_name})
            data_counter.append({'value': c[2] / budget, 'type': cols[2], 'method': remedy_name})
            data_counter.append({'value': c[3] / budget, 'type': cols[3], 'method': remedy_name})
            data_counter.append({'value': c[4] / budget, 'type': cols[4], 'method': remedy_name})

    df_counter = pd.DataFrame(data_counter)

    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/counters/png").mkdir(parents=True,
                                                                                            exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/counters/pdf").mkdir(parents=True,
                                                                                            exist_ok=True)
    plotCounter(data=df_counter,
                dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/counters",
                filename=f"{fitness}_{D}D_counters",
                xlabel="Type of exploration",
                ylabel="Number of fitness evaluations",
                title="Classification of particles exploration")

    R = 0
    with open(
            f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_PSO.pickle",
            "rb") as f:
        particles_counters_PSO = pickle.load(f)
    with open(
            f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_FSTPSO.pickle",
            "rb") as f:
        particles_counters_FSTPSO = pickle.load(f)
    with open(
            f"{dir_baseline}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_RINGPSO.pickle",
            "rb") as f:
        particles_counters_RINGPSO = pickle.load(f)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration/pdf").mkdir(parents=True,
                                                                                               exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration/png").mkdir(parents=True,
                                                                                               exist_ok=True)
    plotParticlesCounters(data=particles_counters_PSO,
                          dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration",
                          filename=f"{fitness}_{D}D_{R}R_PC_PSO")
    plotParticlesCounters(data=particles_counters_FSTPSO,
                          dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration",
                          filename=f"{fitness}_{D}D_{R}R_PC_FSTPSO")
    plotParticlesCounters(data=particles_counters_RINGPSO,
                          dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration",
                          filename=f"{fitness}_{D}D_{R}R_PC_RINGPSO")
    for remedy_name in remedy_names:
        fp = f"results/{remedy_name}/{fitness}/{budget_str}/Counters/{fitness}_{D}D_{R}R_particles_counters_{remedy_name}.pickle"
        with open(fp, "rb") as f:
            particles_counters_remedy = pickle.load(f)
            fp_c = f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/exploration"
            plotParticlesCounters(data=particles_counters_remedy,
                                  dir_name=fp_c,
                                  filename=f"{fitness}_{D}D_{R}R_PC_{remedy_name}")

    ###
    basins_PSO_cum = {}
    basins_FSTPSO_cum = {}
    basins_RINGPSO_cum = {}
    basins_PSO = {}
    basins_FSTPSO = {}
    basins_RINGPSO = {}

    basins_remedies = []
    basins_remedies_cum = []

    for R in runs:
        with open(f"{dir_baseline}/{fitness}/{budget_str}/populations/{fitness}_{D}D_{R}R_population.pickle",
                  "rb") as f:
            initial_population = pickle.load(f)
        with open(
                f"{dir_baseline}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_PSO.pickle",
                "rb") as f:
            basins_pickle = pickle.load(f)
            # basins_pickle.insert(0, set([tuple(np.round(X) + np.zeros(args.D)) for X in initial_population]))
            tmpbasins_cum = basins_to_count_cum(basins_pickle)
            basins_PSO_cum[R] = copy.deepcopy(tmpbasins_cum)
            tmpbasins = basins_to_count(basins_pickle)
            basins_PSO[R] = copy.deepcopy(tmpbasins)
        with open(
                f"{dir_baseline}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_FSTPSO.pickle",
                "rb") as f:
            basins_pickle = pickle.load(f)
            basins_pickle.insert(0, set([tuple(np.round(X) + np.zeros(args.D)) for X in initial_population]))
            tmpbasins_cum = basins_to_count_cum(basins_pickle)
            basins_FSTPSO_cum[R] = copy.deepcopy(tmpbasins_cum)
            tmpbasins = basins_to_count(basins_pickle)
            basins_FSTPSO[R] = copy.deepcopy(tmpbasins)
        with open(
                f"{dir_baseline}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_RINGPSO.pickle",
                "rb") as f:
            basins_pickle = pickle.load(f)
            tmpbasins_cum = basins_to_count_cum(basins_pickle)
            basins_RINGPSO_cum[R] = copy.deepcopy(tmpbasins_cum)
            tmpbasins = basins_to_count(basins_pickle)
            basins_RINGPSO[R] = copy.deepcopy(tmpbasins)

    for remedy_name in remedy_names:
        basins_remedy_cum = []
        basins_remedy = []
        for R in runs:
            fp = f"results/{remedy_name}/{fitness}/{budget_str}/Basins/{fitness}_{D}D_{R}R_basins_iters_{remedy_name}.pickle"
            with open(fp, "rb") as f:
                basins_pickle = pickle.load(f)
                tmpbasins_cum = basins_to_count_cum(basins_pickle)
                basins_remedy_cum.append(copy.deepcopy(tmpbasins_cum))
                tmpbasins = basins_to_count(basins_pickle)
                basins_remedy.append(tmpbasins)
        basins_remedies_cum.append(copy.deepcopy(basins_remedy_cum))
        basins_remedies.append(copy.deepcopy(basins_remedy))

    obj_basins_remedies_cum = []
    obj_basins_remedies = []

    basins_cum_to_boxplot = []

    for R in runs:
        basins_cum_to_boxplot.append({'method': remedy_name_label_map['pso'], 'n': (basins_PSO_cum[R])[-1]})
        basins_cum_to_boxplot.append({'method': remedy_name_label_map['ringpso'], 'n': (basins_RINGPSO_cum[R])[-1]})
        basins_cum_to_boxplot.append({'method': remedy_name_label_map['fstpso'], 'n': (basins_FSTPSO_cum[R])[-1]})

    for remedy_id in range(len(remedy_names)):
        obj_basins_remedy = {}
        obj_basins_remedy_cum = {}
        for R in runs:
            obj_basins_remedy_cum[R] = basins_remedies_cum[remedy_id][R]
            obj_basins_remedy[R] = basins_remedies[remedy_id][R]
            basins_cum_to_boxplot.append(
                {
                    'method': remedy_name_label_map[remedy_names[remedy_id]],
                    'n': basins_remedies_cum[remedy_id][R][-1]
                }
            )

        obj_basins_remedies_cum.append(obj_basins_remedy_cum)
        obj_basins_remedies.append(obj_basins_remedy)

    df_basins_PSO_cum = pd.DataFrame(basins_PSO_cum)
    df_basins_FSTPSO_cum = pd.DataFrame(basins_FSTPSO_cum)
    df_basins_RINGPSO_cum = pd.DataFrame(basins_RINGPSO_cum)

    basins_PSO_cum_median = df_basins_PSO_cum.median(axis=1).values.tolist()
    basins_FSTPSO_cum_median = df_basins_FSTPSO_cum.median(axis=1).values.tolist()
    basins_RINGPSO_cum_median = df_basins_RINGPSO_cum.median(axis=1).values.tolist()

    df_basins_PSO = pd.DataFrame(basins_PSO)
    df_basins_FSTPSO = pd.DataFrame(basins_FSTPSO)
    df_basins_RINGPSO = pd.DataFrame(basins_RINGPSO)

    basins_PSO_median = df_basins_PSO.median(axis=1).values.tolist()
    basins_FSTPSO_median = df_basins_FSTPSO.median(axis=1).values.tolist()
    basins_RINGPSO_median = df_basins_RINGPSO.median(axis=1).values.tolist()

    basins_remedies_median = []
    basins_remedies_cum_median = []

    for remedy_id in range(len(remedy_names)):
        df_basins_remedy_cum = pd.DataFrame(obj_basins_remedies_cum[remedy_id])
        basins_remedy_cum_median = df_basins_remedy_cum.median(axis=1).values.tolist()
        basins_remedies_cum_median.append(basins_remedy_cum_median)
        df_basins_remedy = pd.DataFrame(obj_basins_remedies[remedy_id])
        basins_remedy_median = df_basins_remedy.median(axis=1).values.tolist()
        basins_remedies_median.append(basins_remedy_median)

    print("lens")
    print(len(basins_PSO_cum_median), len(basins_RINGPSO_cum_median), len(basins_FSTPSO_cum_median))
    for brcm in basins_remedies_cum_median:
        print(len(brcm))
    basins_cum_to_plot = ([basins_PSO_cum_median, basins_RINGPSO_cum_median, basins_FSTPSO_cum_median] +
                          basins_remedies_cum_median)
    basins_to_plot = [basins_PSO_median, basins_RINGPSO_median, basins_FSTPSO_median] + basins_remedies_median
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/basins/pdf").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/basins/png").mkdir(parents=True, exist_ok=True)
    plotBasins(
        basins_cum_to_plot,
        labels_to_plot,
        "Cumulative N of explored basins",
        dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/basins",
        filename=f"{fitness}_{D}D_new_basins",
        labels_legend=remedy_name_label_map.values()
    )

    plotBasins(
        basins_to_plot,
        labels_to_plot,
        "N of explored basins per iter",
        dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/basins",
        filename=f"{fitness}_{D}D_basins_iter",
        labels_legend=remedy_name_label_map.values()
    )

    df_basins = pd.DataFrame(basins_cum_to_boxplot)
    plot_boxplot(
        data=df_basins,
        dir_name=f"{dir_results_base}/plots/{fitness}/{budget_str}/{D}/basins",
        filename=f"{fitness}_{D}D_basins_cum_boxplot",
        function_name=fitness,
        x_name="method",
        y_name="n",
        xlabel="Method",
        ylabel="Number of explored basins",
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
    parser.add_argument("--function", '-F', dest='fitness', type=str, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int, default=2)
    parser.add_argument("--budget", '-B', dest='budget_str', type=str, default="4B")
    args = parser.parse_args()

    remedy_names = [
        "pso_ring_remedy_1",
        "pso_ring_remedy_2",
        "pso_ring_remedy_3",
        "pso_ring_remedy_4",
        "fstpso_remedy_1",
        "fstpso_remedy_2",
        "fstpso_remedy_3",
        "fstpso_remedy_4",
        "stufstpso",
    ]
    remedy_name_label_map = {
        "pso": "PSO",
        "ringpso": "RING-PSO",
        "fstpso": "FST-PSO",
        "pso_ring_remedy_1": "RING-PSO R1",
        "pso_ring_remedy_2": "RING-PSO R2",
        "pso_ring_remedy_3": "RING-PSO R3",
        "pso_ring_remedy_4": "RING-PSO R4",
        "fstpso_remedy_1": "FST-PSO R1",
        "fstpso_remedy_2": "FST-PSO R2",
        "fstpso_remedy_3": "FST-PSO R3",
        "fstpso_remedy_4": "FST-PSO R4",
        "stufstpso": "STU-PSO",
    }

    dir_results_base = "results/remedies"
    pathlib.Path(f"{dir_results_base}/plots/{args.fitness}/{args.budget_str}").mkdir(parents=True, exist_ok=True)

    main(D=args.D, fitness=args.fitness, budget_str=args.budget_str, budget=args.D * 1e4,
         dir_results_base=dir_results_base, remedy_names=remedy_names, remedy_name_label_map=remedy_name_label_map)
    mainVelocities(D=args.D, fitness=args.fitness, budget_str=args.budget_str, budget=args.D * 1e4,
                   dir_results_base=dir_results_base, remedy_names=remedy_names,
                   remedy_name_label_map=remedy_name_label_map)
