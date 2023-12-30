import sys
import copy
import functools
import pathlib
import argparse
import math
import pickle

sys.path.insert(0, "code")

from benchmarks.soo.limits import limits
from benchmarks.soo.functions import functions
import numpy as np

from pso_ring_remedy_1 import PSO_ring_remedy_1
from pso_ring_remedy_2 import PSO_ring_remedy_2
from pso_ring_remedy_3 import PSO_ring_remedy_3
from pso_ring_remedy_4 import PSO_ring_remedy_4
from fstpso_remedy_1 import FuzzyPSO_remedy_1
from fstpso_remedy_2 import FuzzyPSO_remedy_2
from fstpso_remedy_3 import FuzzyPSO_remedy_3
from fstpso_remedy_4 import FuzzyPSO_remedy_4


remedy_map = {
    "pso_ring_remedy_1": PSO_ring_remedy_1,
    "pso_ring_remedy_2": PSO_ring_remedy_2,
    "pso_ring_remedy_3": PSO_ring_remedy_3,
    "pso_ring_remedy_4": PSO_ring_remedy_4,
    "fstpso_remedy_1": FuzzyPSO_remedy_1,
    "fstpso_remedy_2": FuzzyPSO_remedy_2,
    "fstpso_remedy_3": FuzzyPSO_remedy_3,
    "fstpso_remedy_4": FuzzyPSO_remedy_4,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=str, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int)
    parser.add_argument("--run", '-R', dest='R', type=int)
    parser.add_argument("--remedy", '-RN', dest='remedy_name', type=str, default="pso_ring_remedy_1")

    args = parser.parse_args()

    remedy_name = args.remedy_name
    dir_results_base = f'results/{remedy_name}'
    budget = 1e4 * args.D
    budget_str = "4B"

    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/plots/{args.fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_{args.remedy_name}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Counters').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Basins').mkdir(parents=True, exist_ok=True)

    search_space = [limits[args.fitness]] * args.D
    f = functions[args.fitness]

    global_best_per_iter = []
    curr_sum = 0
    velocities_iter = []
    cum_avg = []
    save_initial_population = True
    populations = []

    # 0 -> Successful Exploration
    # 1 -> Deceptive Exploration
    # 2 -> Failed Exploration
    # 3 -> Successful Rejection
    # 4 -> Exploitation

    counter_exploration = [0, 0, 0, 0, 0]
    particles_counters = []
    basins_iters = []

    def callback(s, curr_sum_f=curr_sum):
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        n_particles = len(particles)
        curr_i = s.Iterations
        particles_counters.append([0 for _ in s.Solutions])
        # print(f"+current iteration: {curr_i}")

        velocities_curr = [np.linalg.norm(p.V) for p in particles]
        curr_sum_f = curr_sum_f + np.sum(velocities_curr)
        cum_avg.append(curr_sum_f / (n_particles * curr_i))
        velocities_iter.append(velocities_curr)

        basins_iters.append(set([tuple(np.round(p.X)+np.zeros(args.D)) for p in particles]))

        for i_p in range(len(particles)):
            p = particles[i_p]
            X = p.X
            basin_X = np.round(X)

            prevB = p.PreviousPbest
            basin_prevB = np.round(prevB)
            f_X = f(X)
            f_basin_X = f(basin_X)
            f_prevB = f(prevB)
            f_basin_prevB = f(basin_prevB)
            if np.all(basin_X == basin_prevB):
                counter_exploration[4] += 1  # Exploitation
                particles_counters[curr_i - 1][i_p] = 4
            else:
                if f_X < f_prevB:
                    if f_basin_X <= f_basin_prevB:
                        counter_exploration[0] += 1  # 0 -> Successful Exploration
                        particles_counters[curr_i - 1][i_p] = 0
                    elif f_basin_X > f_basin_prevB:
                        counter_exploration[1] += 1  # 1 -> Deceptive Exploration
                        particles_counters[curr_i - 1][i_p] = 1
                elif f_X >= f_prevB:
                    if f_basin_X < f_basin_prevB:
                        counter_exploration[2] += 1  # 2 -> Failed Exploration
                        particles_counters[curr_i - 1][i_p] = 2
                    elif f_basin_X >= f_basin_prevB:
                        counter_exploration[3] += 1  # 3 -> Successful Rejection
                        particles_counters[curr_i - 1][i_p] = 3

    initial_population = []

    with open(
            f'results/vanilla_50P/{args.fitness}/{budget_str}/populations/{args.fitness}_{args.D}D_{args.R}R_population.pickle',
            'rb') as f_ip:
        initial_population = pickle.load(f_ip)

    numberofparticles = len(initial_population)
    fp_init = remedy_map[args.remedy_name]
    FP = fp_init()
    FP.Boundaries = search_space
    FP.FITNESS = f

    """NFEcur = 0
    curpop = 0
    SEQ = []
    curpop = numberofparticles
    NFEcur = curpop
    SEQ.append(curpop)
    while (1):
        if NFEcur + curpop > budget:
            curpop = budget - NFEcur
        SEQ.append(curpop)
        NFEcur += curpop
        if NFEcur >= budget:
            break
    est_iterations = len(SEQ) - 1"""
    est_iterations = int((budget-numberofparticles)/numberofparticles) - 1

    if "fst" in remedy_name:
        FP.set_fitness(f)
        FP.set_search_space(search_space)
        FP.set_swarm_size(numberofparticles)
        FP.NumberOfParticles = numberofparticles
    else:
        FP.set_number_of_particles(numberofparticles)
    FP.Dimensions = args.D
    FP.MaxIterations = est_iterations

    callback_partial = functools.partial(callback)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }
    print(remedy_name)
    if "fst" in remedy_name:
        FP.solve_with_fstpso(callback=callback_param, initial_guess_list=initial_population, max_FEs=budget, max_iter=est_iterations)
    else:
        FP.Solve(None, callback=callback_param, initial_guess_list=initial_population)

    counter_exploration_remedy = copy.deepcopy(counter_exploration)
    particles_counters_remedy = copy.deepcopy(particles_counters)
    gbest_0 = np.min([f(s) for s in initial_population])
    # gbest initial population
    global_best_per_iter.insert(0, gbest_0)
    gbest_remedy = copy.deepcopy(global_best_per_iter)

    velocities_iter_remedy = copy.deepcopy(velocities_iter)
    # basins initial population
    basins_to_add = set([tuple(np.round(X) + np.zeros(args.D)) for X in initial_population])
    basins_iters.insert(0, basins_to_add)
    basins_iters_remedy = copy.deepcopy(basins_iters)

    print(f"best, {remedy_name}: {gbest_remedy[-1]}")
    print(f"lens, {remedy_name}: {len(gbest_remedy)}")
    print(f"lens, {remedy_name}: {len(basins_iters_remedy)}")

    # dump gbest per iter
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_{remedy_name}/{args.fitness}_{args.D}D_{args.R}R_gbest_{remedy_name}.txt',
            "w") as f:
        for i in gbest_remedy:
            f.write(str(i) + "\n")

    # dump velocities
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_{remedy_name}.pickle',
            'wb') as f:
        pickle.dump(velocities_iter_remedy, f)
    # dump counters exploration
    with open(
            f"{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_{remedy_name}.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_remedy, f)
    # dump particles counters
        with open(
                f'{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_{remedy_name}.pickle',
                'wb') as f:
            pickle.dump(particles_counters_remedy, f)
    # dump basins per iter
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_{remedy_name}.pickle',
            'wb') as f:
        pickle.dump(basins_iters_remedy, f)
