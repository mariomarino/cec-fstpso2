import sys
import copy
import functools
import pathlib
import argparse
import math
import pickle

sys.path.insert(0, "code")
sys.path.insert(0, "code/methods")

from benchmarks.soo.limits import limits
from benchmarks.soo.functions import functions
import numpy as np

from fstpso_original import FuzzyPSO
from pso_ring import PSO_ring


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--fitness", required=True)
    parser.add_argument("-D", "--dimensions", required=True)
    parser.add_argument("-R", "--run", required=True)

    args = vars(parser.parse_args())  # parser.parse_args()
    
    fitness = f"CEC17-F{args['fitness']}"
    run = int(args['run'])
    dim = int(args['dimensions'])

    print(f"{fitness}, {dim}D, {run}R")

    dir_results_base = f'results_CEC'

    budget = 1e4 * dim
    budget_str = "4B"

    search_space = [limits[fitness]] * dim
    f = functions[fitness]

    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/populations').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/gbest_RINGPSO').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/gbest_FSTPSO').mkdir(parents=True, exist_ok=True)
    """pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/Velocities_RINGPSO').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/Velocities_FSTPSO').mkdir(parents=True, exist_ok=True)"""



    numberofparticles = 50

    global_best_per_iter = []
    """curr_sum = 0
    velocities_iter = []
    cum_avg = []"""
    save_initial_population = True
    populations = []

    def callback_RINGPSO(s, populations_f=None):  # , curr_sum_f=curr_sum,
        if populations_f is None:
            populations_f = populations
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        n_particles = len(particles)
        curr_i = s.Iterations
        populations_f.append(copy.deepcopy([p.X for p in s.Solutions]))

        """velocities_curr = [np.linalg.norm(p.V) for p in particles]
        curr_sum_f = curr_sum_f + np.sum(velocities_curr)
        cum_avg.append(curr_sum_f / (n_particles * curr_i))
        velocities_iter.append(velocities_curr)"""


    def callback_FSTPSO(s):  # , curr_sum_f=curr_sum
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        n_particles = len(particles)
        curr_i = s.Iterations

        """velocities_curr = [np.linalg.norm(p.V) for p in particles]
        curr_sum_f = curr_sum_f + np.sum(velocities_curr)
        cum_avg.append(curr_sum_f / (n_particles * curr_i))
        velocities_iter.append(velocities_curr)"""

    callback_partial = functools.partial(callback_RINGPSO, populations_f=populations)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }

    FP = PSO_ring()
    FP.Boundaries = search_space
    FP.FITNESS = f
    est_iterations = int((budget - numberofparticles) / numberofparticles) - 1

    FP.MaxIterations = est_iterations
    FP.set_number_of_particles(numberofparticles)
    FP.Dimensions = dim
    print("RINGPSO")
    FP.Solve(None, callback=callback_param)

    print(f"iters: {FP.MaxIterations}")
    initial_population = copy.deepcopy(populations[0])
    print(f"population size: {len(initial_population)}")
    # dump initial population
    with open(
            f'{dir_results_base}/{fitness}/{budget_str}/populations/{fitness}_{dim}D_{run}R_population.pickle',
            'wb') as fdump:
        pickle.dump(initial_population, fdump)
    del populations
    populations = []
    # gbest initial population
    gbest_0 = np.min([f(s) for s in initial_population])
    global_best_per_iter.insert(0, gbest_0)
    gbest_RINGPSO = copy.deepcopy(global_best_per_iter)
    """velocities_iter_RINGPSO = copy.deepcopy(velocities_iter)
    del velocities_iter
    velocities_iter = []
    curr_sum = 0"""

    FP = FuzzyPSO()
    FP.set_fitness(f)
    FP.set_search_space(search_space)
    FP.MaxIterations = est_iterations
    global_best_per_iter = []
    callback_partial = functools.partial(callback_FSTPSO)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }
    FP.set_swarm_size(numberofparticles)

    print("FSTPSO")
    FP.solve_with_fstpso(
        callback=callback_param,
        initial_guess_list=initial_population,
        max_FEs=budget,
        max_iter=est_iterations
    )
    print(f"iters: {FP.MaxIterations}")
    global_best_per_iter.insert(0, gbest_0)
    gbest_FSTPSO = copy.deepcopy(global_best_per_iter)

    """velocities_iter_FSTPSO = copy.deepcopy(velocities_iter)

    del velocities_iter"""
    velocities_iter = []
    global_best_per_iter = []
    curr_sum = 0

    print(f"bests: RINGPSO: {gbest_RINGPSO[-1]}, FSTPSO: {gbest_FSTPSO[-1]}")
    print(f"first: RINGPSO: {gbest_RINGPSO[0]}, FSTPSO: {gbest_FSTPSO[0]}")
    print(f"second: RINGPSO: {gbest_RINGPSO[1]}, FSTPSO: {gbest_FSTPSO[1]}")
    print(f"lens: RINGPSO: {len(gbest_RINGPSO)}, FSTPSO: {len(gbest_FSTPSO)}")


    # dump gbest per iter
    with open(
            f'{dir_results_base}/{fitness}/{budget_str}/gbest_RINGPSO/{fitness}_{dim}D_{run}R_gbest_RINGPSO.txt',
            "w") as f:
        for i in gbest_RINGPSO:
            f.write(str(i) + "\n")
    with open(
            f'{dir_results_base}/{fitness}/{budget_str}/gbest_FSTPSO/{fitness}_{dim}D_{run}R_gbest_FSTPSO.txt',
            "w") as f:
        for i in gbest_FSTPSO:
            f.write(str(i) + "\n")
    if False:
        with open(
                f'{dir_results_base}/{fitness}/{budget_str}/Velocities_RINGPSO/{fitness}_{dim}D_{run}R_velocities_RINGPSO.pickle',
                'wb') as f:
            pickle.dump(velocities_iter_RINGPSO, f)
        with open(
                f'{dir_results_base}/{fitness}/{budget_str}/Velocities_FSTPSO/{fitness}_{dim}D_{run}R_velocities_FSTPSO.pickle',
                'wb') as f:
            pickle.dump(velocities_iter_FSTPSO, f)


