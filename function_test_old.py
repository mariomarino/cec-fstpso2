import sys
import copy
import functools
import pathlib
import argparse
import math
import pickle
# sys.path.insert(0,"/home/mario/projects/cec-fstpso")
sys.path.insert(0,"/home/mario/projects/cec-fstpso/code")

from benchmarks.soo.limits import limits
from benchmarks.soo.functions import functions
import numpy as np
# from fstpso import FuzzyPSO
from fstpso_original import FuzzyPSO

from fstpso_custom import StallFuzzyPSO
from fstpso_stu import StuFuzzyPSO
from fstpso_original import PSO_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=str, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int)
    parser.add_argument("--run", '-R', dest='R', type=int)
    args = parser.parse_args()

    dir_results_base = "../results/results_counters_stu"

    budget = 1e4 * args.D
    budget_str = "4B"

    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/populations").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/plots/{args.fitness}/{budget_str}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/gbest_nostall").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/gbest_stall").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/DeltaVelocity").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/Velocities").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/Counters").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"./{dir_results_base}/{args.fitness}/{budget_str}/Basins").mkdir(parents=True, exist_ok=True)

    search_space = [limits[args.fitness]] * args.D

    global_best_per_iter = []
    curr_sum = 0
    velocities_iter = []
    cum_avg = []
    save_initial_population = True
    populations = []

    Social = []
    Cognitive = []
    Inertia = []

    """counter_exploration = {
        "SE": 0,
        "SR": 0,
        "DE": 0,
        "FE": 0
    }"""

    # 0 -> Successful Exploration
    # 1 -> Deceptive Exploration
    # 2 -> Failed Exploration
    # 3 -> Successful Rejection

    counter_exploration = [0, 0, 0, 0, 0]
    particles_counters = []
    basins_iters = []

    f = functions[args.fitness]


    def callback(s, curr_sum_f=curr_sum, populations_f=None,
                 Social_f=None, Cognitive_f=None, Inertia_f=None, StallDetection=None):
        if Inertia_f is None:
            Inertia_f = Inertia
        if Cognitive_f is None:
            Cognitive_f = Cognitive
        if Social_f is None:
            Social_f = Social
        if populations_f is None:
            populations_f = populations
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        N = len(particles)
        curr_i = s.Iterations
        particles_counters.append([0 for _ in s.Solutions])
        populations_f.append(copy.deepcopy([p.X for p in s.Solutions]))

        velocities_curr = [np.linalg.norm(p.V) for p in particles]
        curr_sum_f = curr_sum_f + np.sum(velocities_curr)
        cum_avg.append(curr_sum_f / (N * curr_i))
        velocities_iter.append(velocities_curr)

        Social_f.append([p.SocialFactor for p in particles])
        Cognitive_f.append([p.CognitiveFactor for p in particles])
        Inertia_f.append([p.Inertia for p in particles])

        basins_iters.append([])
        for i_p in range(len(particles)):
            p = particles[i_p]
            """B = p.B
            basin_B = np.round(B)"""
            X = p.X
            basin_X = np.round(X)

            basins_iters[curr_i-1].append(copy.deepcopy(basin_X))

            prevB = p.PreviousPbest
            basin_prevB = np.round(prevB)
            """f_B = f(B)
            f_basin_B = f(basin_B)"""
            f_X = f(X)
            f_basin_X = f(basin_X)
            f_prevB = f(prevB)
            f_basin_prevB = f(basin_prevB)
            if np.all(basin_X == basin_prevB):
                counter_exploration[4] += 1     # Exploitation
                particles_counters[curr_i - 1][i_p] = 4
            else:
                if f_X < f_prevB:
                    if f_basin_X < f_basin_prevB:
                        counter_exploration[0] += 1  # 0 -> Successful Exploration
                        particles_counters[curr_i-1][i_p] = 0
                    elif f_basin_X > f_basin_prevB:
                        counter_exploration[1] += 1  # 1 -> Deceptive Exploration
                        particles_counters[curr_i - 1][i_p] = 1
                    else:
                        counter_exploration[4] += 1
                        particles_counters[curr_i - 1][i_p] = 4
                elif f_X > f_prevB:
                    if f_basin_X < f_basin_prevB:
                        counter_exploration[2] += 1  # 2 -> Failed Exploration
                        particles_counters[curr_i - 1][i_p] = 2
                    elif f_basin_X > f_basin_prevB:
                        counter_exploration[3] += 1  # 3 -> Successful Rejection
                        particles_counters[curr_i - 1][i_p] = 3
                    else:
                        counter_exploration[4] += 1
                        particles_counters[curr_i - 1][i_p] = 4
                else:
                    counter_exploration[4] += 1
                    particles_counters[curr_i - 1][i_p] = 4



    """callback_param = {
        'interval': 1,
        'function': callback
    }"""

    numberofparticles = int(10 + 2 * math.sqrt(args.D))

    # run vanilla
    # save initial population
    FP = FuzzyPSO()
    FP = PSO_new()
    print("here")
    FP.set_fitness(f)
    FP.set_search_space(search_space)
    # population = copy.deepcopy([p.X for p in FP.Solutions])
    callback_partial = functools.partial(callback, populations_f=populations, Social_f=Social, Cognitive_f=Cognitive,
                                         Inertia_f=Inertia)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }

    FP.solve_with_fstpso(callback=callback_param, max_FEs=budget)

    counter_exploration_nostall = copy.deepcopy(counter_exploration)
    particles_counters_nostall = copy.deepcopy(particles_counters)

    Social_r = copy.deepcopy(Social)
    Cognitive_r = copy.deepcopy(Cognitive)
    Inertia_r = copy.deepcopy(Inertia)

    gbest_nostall = copy.deepcopy(global_best_per_iter)
    initial_population = copy.deepcopy(populations[0])
    del populations
    populations = []
    del Social
    del Cognitive
    del Inertia
    Social = []
    Cognitive = []
    Inertia = []
    velocities_iter_nostall = copy.deepcopy(velocities_iter)
    del velocities_iter
    velocities_iter = []
    del counter_exploration
    counter_exploration = [0, 0, 0, 0, 0]
    del particles_counters
    particles_counters = []
    basins_iters_nostall = copy.deepcopy(basins_iters)
    del basins_iters
    basins_iters = []
    curr_sum = 0

    # run stall detection and perturbation
    # load previous population
    # FP = StallFuzzyPSO(SigmaPerc=0.05)  # int(math.log(budget))
    FP = StuFuzzyPSO()
    FP.set_fitness(f)
    FP.set_search_space(search_space)
    global_best_per_iter = []
    KappaMax = math.log(budget)  # 3
    callback_partial = functools.partial(callback, populations_f=populations, Social_f=Social, Cognitive_f=Cognitive,
                                         Inertia_f=Inertia)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }
    # gbest initial population
    gbest_0 = np.max([f(s) for s in initial_population])
    gbest_nostall.insert(0, gbest_0)
    FP.solve_with_fstpso(callback=callback_param, initial_guess_list=initial_population, max_FEs=budget,
                         KappaMax=KappaMax)

    counter_exploration_stall = copy.deepcopy(counter_exploration)
    particles_counters_stall = copy.deepcopy(particles_counters)

    gbest_stall = copy.deepcopy(global_best_per_iter)
    # gbest initial population
    gbest_stall.insert(0, gbest_0)

    DeltaVelocity = FP.DeltaVelocity
    Social_p = copy.deepcopy(Social)
    Cognitive_p = copy.deepcopy(Cognitive)
    Inertia_p = copy.deepcopy(Inertia)

    velocities_iter_stall = copy.deepcopy(velocities_iter)

    basins_iters_stall = copy.deepcopy(basins_iters)

    # dump initial population
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/populations/{args.fitness}_{args.D}D_{args.R}R_population.pickle",
            "wb") as f:
        pickle.dump(initial_population, f)
    # dump gbest no stall
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/gbest_nostall/{args.fitness}_{args.D}D_{args.R}R_gbest_nostall.txt",
            "w") as f:
        for i in gbest_nostall:
            f.write(str(i) + "\n")
    # dump gbest stall
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/gbest_stall/{args.fitness}_{args.D}D_{args.R}R_gbest_stall.txt",
            "w") as f:
        for i in gbest_stall:
            f.write(str(i) + "\n")
    # dump delta velocity
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/DeltaVelocity/{args.fitness}_{args.D}D_{args.R}R_DeltaVelocity.pickle",
            "wb") as f:
        pickle.dump(DeltaVelocity, f)
    # dump reference factors
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Social_r.pickle",
            "wb") as f:
        pickle.dump(Social_r, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Cognitive_r.pickle",
            "wb") as f:
        pickle.dump(Cognitive_r, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Inertia_r.pickle",
            "wb") as f:
        pickle.dump(Inertia_r, f)
    # dump factors after perturb
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Social_p.pickle",
            "wb") as f:
        pickle.dump(Social_p, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Cognitive_p.pickle",
            "wb") as f:
        pickle.dump(Cognitive_p, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Factors/{args.fitness}_{args.D}D_{args.R}R_Inertia_p.pickle",
            "wb") as f:
        pickle.dump(Inertia_p, f)
    # dump velocities
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_nostall.pickle",
            "wb") as f:
        pickle.dump(velocities_iter_nostall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_stall.pickle",
            "wb") as f:
        pickle.dump(velocities_iter_stall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_nostall.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_nostall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_stall.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_stall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_nostall.pickle",
            "wb") as f:
        pickle.dump(particles_counters_nostall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_stall.pickle",
            "wb") as f:
        pickle.dump(particles_counters_stall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_nostall.pickle",
            "wb") as f:
        pickle.dump(basins_iters_nostall, f)
    with open(
            f"./{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_stall.pickle",
            "wb") as f:
        pickle.dump(basins_iters_stall, f)
