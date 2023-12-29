import sys
import copy
import functools
import pathlib
import argparse
import math
import pickle

# sys.path.insert(0,"/home/mario/projects/cec-fstpso")
sys.path.insert(0, "/home/mario/projects/cec-fstpso/code")

from benchmarks.soo.limits import limits
from benchmarks.soo.functions import functions
import numpy as np
# from fstpso import FuzzyPSO
from fstpso_original import FuzzyPSO

from fstpso_custom import StallFuzzyPSO
from fstpso_stu import StuFuzzyPSO
from pso import PSO_new
from pso_ring import PSO_ring


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=str, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int)
    parser.add_argument("--run", '-R', dest='R', type=int)
    # parser.add_argument("--psotype" '-T', dest='PSOTYPE', type=str, default='PSO')
    args = parser.parse_args()

    dir_results_base = f'results/vanilla_50P'

    budget = 1e4 * args.D
    budget_str = "4B"

    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/populations').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/plots/{args.fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_PSO').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_FSTPSO').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_RINGPSO').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/DeltaVelocity').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Factors').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Counters').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{args.fitness}/{budget_str}/Basins').mkdir(parents=True, exist_ok=True)

    search_space = [limits[args.fitness]] * args.D

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

    counter_exploration = [0, 0, 0, 0, 0]
    particles_counters = []
    basins_iters = []

    f = functions[args.fitness]


    def callback():
        pass


    numberofparticles = 50  # int(10 + 2 * math.sqrt(args.D))

    # run vanilla
    # save initial population
    FP = PSO_new()

    def callback_PSO(s, curr_sum_f=curr_sum, populations_f=None):
        if populations_f is None:
            populations_f = populations
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        n_particles = len(particles)
        curr_i = s.Iterations
        particles_counters.append([0 for _ in s.Solutions])
        populations_f.append(copy.deepcopy([p.X for p in s.Solutions]))

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

    def callback_FSTPSO(s, curr_sum_f=curr_sum):
        global_best_per_iter.append(s.G.CalculatedFitness)
        particles = s.Solutions
        n_particles = len(particles)
        curr_i = s.Iterations
        particles_counters.append([0 for _ in s.Solutions])

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


    # population = copy.deepcopy([p.X for p in FP.Solutions])
    callback_partial = functools.partial(callback_PSO, populations_f=populations)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }

    FP.Boundaries = search_space
    FP.FITNESS = f

    NFEcur = 0
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
    est_iterations = len(SEQ) - 2

    FP.MaxIterations = est_iterations
    FP.set_number_of_particles(numberofparticles)
    FP.Dimensions = args.D
    print("PSO")
    FP.Solve(None, callback=callback_param)
    print(f"iters: {FP.MaxIterations}")
    counter_exploration_PSO = copy.deepcopy(counter_exploration)
    particles_counters_PSO = copy.deepcopy(particles_counters)

    gbest_PSO = copy.deepcopy(global_best_per_iter)
    initial_population = copy.deepcopy(populations[0])
    del populations
    populations = []
    velocities_iter_PSO = copy.deepcopy(velocities_iter)
    del counter_exploration
    counter_exploration = [0, 0, 0, 0, 0]
    del velocities_iter
    velocities_iter = []
    del particles_counters
    particles_counters = []
    basins_iters_PSO = copy.deepcopy(basins_iters)
    del basins_iters
    basins_iters = []
    curr_sum = 0

    # run stall detection and perturbation
    # load previous population
    # FP = StallFuzzyPSO(SigmaPerc=0.05)  # int(math.log(budget))
    # FSTPSO
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
    # gbest initial population
    gbest_0 = np.min([f(s) for s in initial_population])  # ] * numberofparticles
    # gbest_PSO.insert(0, gbest_0)
    print("FSTPSO")
    FP.solve_with_fstpso(callback=callback_param, initial_guess_list=initial_population, max_FEs=budget)
    print(f"iters: {FP.MaxIterations}")
    basins_iters.insert(0, set([tuple(np.round(X) + np.zeros(args.D)) for X in initial_population]))
    counter_exploration_FSTPSO = copy.deepcopy(counter_exploration)
    particles_counters_FSTPSO = copy.deepcopy(particles_counters)

    gbest_FSTPSO = copy.deepcopy(global_best_per_iter)
    # gbest initial population
    gbest_FSTPSO.insert(0, gbest_0)

    velocities_iter_FSTPSO = copy.deepcopy(velocities_iter)

    basins_iters_FSTPSO = copy.deepcopy(basins_iters)

    del counter_exploration
    counter_exploration = [0, 0, 0, 0, 0]
    del velocities_iter
    velocities_iter = []
    del particles_counters
    particles_counters = []
    del basins_iters
    basins_iters = []
    global_best_per_iter = []
    curr_sum = 0

    callback_partial = functools.partial(callback_PSO)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }

    # ring PSO

    FP = PSO_ring()
    FP.Boundaries = search_space
    FP.FITNESS = f
    FP.MaxIterations = est_iterations
    FP.Dimensions = args.D
    FP.set_number_of_particles(numberofparticles)
    print("RINGPSO")
    FP.Solve(None, callback=callback_param, initial_guess_list=initial_population)
    print(f"iters: {FP.MaxIterations}")
    counter_exploration_RINGPSO = copy.deepcopy(counter_exploration)
    particles_counters_RINGPSO = copy.deepcopy(particles_counters)

    gbest_RINGPSO = copy.deepcopy(global_best_per_iter)

    print(f"bests: PSO: {gbest_PSO[0]}, FSTPSO: {gbest_FSTPSO[0]}, RINGPSO: {gbest_RINGPSO[0]}")
    print(f"lens: PSO: {len(gbest_PSO)}, FSTPSO: {len(gbest_FSTPSO)}, RINGPSO: {len(gbest_RINGPSO)}")
    # gbest initial population
    # gbest_RINGPSO.insert(0, gbest_0)

    velocities_iter_RINGPSO = copy.deepcopy(velocities_iter)

    basins_iters_RINGPSO = copy.deepcopy(basins_iters)

    # dump initial population
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/populations/{args.fitness}_{args.D}D_{args.R}R_population.pickle',
            'wb') as f:
        pickle.dump(initial_population, f)
    # dump gbest no stall
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_PSO/{args.fitness}_{args.D}D_{args.R}R_gbest_PSO.txt',
            "w") as f:
        for i in gbest_PSO:
            f.write(str(i) + "\n")
    # dump gbest stall
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_FSTPSO/{args.fitness}_{args.D}D_{args.R}R_gbest_FSTPSO.txt',
            "w") as f:
        for i in gbest_FSTPSO:
            f.write(str(i) + "\n")
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/gbest_RINGPSO/{args.fitness}_{args.D}D_{args.R}R_gbest_RINGPSO.txt',
            "w") as f:
        for i in gbest_RINGPSO:
            f.write(str(i) + "\n")

    # dump velocities
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_PSO.pickle',
            'wb') as f:
        pickle.dump(velocities_iter_PSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_FSTPSO.pickle',
            'wb') as f:
        pickle.dump(velocities_iter_FSTPSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Velocities/{args.fitness}_{args.D}D_{args.R}R_velocities_RINGPSO.pickle',
            'wb') as f:
        pickle.dump(velocities_iter_RINGPSO, f)
    with open(
            f"{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_PSO.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_PSO, f)
    with open(
            f"{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_FSTPSO.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_FSTPSO, f)
    with open(
            f"{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_counter_exploration_RINGPSO.pickle",
            "wb") as f:
        pickle.dump(counter_exploration_RINGPSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_PSO.pickle',
            'wb') as f:
        pickle.dump(particles_counters_PSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_FSTPSO.pickle',
            'wb') as f:
        pickle.dump(particles_counters_FSTPSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Counters/{args.fitness}_{args.D}D_{args.R}R_particles_counters_RINGPSO.pickle',
            'wb') as f:
        pickle.dump(particles_counters_RINGPSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_PSO.pickle',
            'wb') as f:
        pickle.dump(basins_iters_PSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_FSTPSO.pickle',
            'wb') as f:
        pickle.dump(basins_iters_FSTPSO, f)
    with open(
            f'{dir_results_base}/{args.fitness}/{budget_str}/Basins/{args.fitness}_{args.D}D_{args.R}R_basins_iters_RINGPSO.pickle',
            'wb') as f:
        pickle.dump(basins_iters_RINGPSO, f)
