import copy
from functools import partial

from code.fstpso_custom import StallFuzzyPSO
from fstpso import FuzzyPSO
import numpy as np
import matplotlib.pyplot as plt
import math


def ackley_ext(x, a=20, b=0.2, c=2 * math.pi):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        # Avoid mono-dimensional points.
        x = x[:, np.newaxis]
    # At this point, x is a matrix where each column is an individual and each row a dimension.
    y = - a * np.exp(-b * np.sqrt(np.mean(x ** 2, axis=0))) - np.exp(np.mean(np.cos(c * x), axis=0)) + a + np.e
    # The axis=0 operation parallels the sum of the matrix directly using an efficient NumPy operation.
    if y.ndim == 1:
        # If the input is a scalar then return a scalar,
        y = np.squeeze(y)
    return y


ackley = partial(ackley_ext, a=20, b=0.2, c=2 * math.pi)

dims = 2
left_extreme, right_extreme = -32, 32

global_best_per_iter = []
curr_sum = 0
velocities_iter = []
cum_avg = []


def callback(s, curr_sum=curr_sum):
    global_best_per_iter.append(s.G.CalculatedFitness)
    particles = s.Solutions
    N = len(particles)
    i = s.Iterations
    velocities_curr = [np.linalg.norm(p.V) for p in particles]
    curr_sum = curr_sum + np.sum(velocities_curr)
    cum_avg.append(curr_sum / (N * i))
    velocities_iter.append(velocities_curr)


callback_param = {
    'interval': 1,
    'function': callback
}

"""FP = FuzzyPSO()  # StallFuzzyPSO()
FP.set_fitness(ackley)
FP.set_search_space([[left_extreme, right_extreme]] * dims)"""

# print(FP)

# result = FP.solve_with_fstpso(callback=callback_param)
# print(result)


def plotVelocities(VelocitiesOverTimeRead, f_name, save=False):
    velocitiesOverTime = []
    previousVelocities = []
    pointsAboveAvg = []

    k = 0
    for i in VelocitiesOverTimeRead:
        k += 1
        previousVelocities = previousVelocities + i
        mean_vel = np.mean(previousVelocities)
        velocitiesOverTime.append(mean_vel)
        pointsAboveAvg.append(len([h for h in i if h < mean_vel]))

    xs = list(range(0, len(VelocitiesOverTimeRead)))
    plt.rcParams["figure.figsize"] = (10, 6)
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
    plt.show()


def plotConvergence(BestOverTime, f_name, save=False):
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.tick_params(axis='y', colors='black')
    plt.plot(BestOverTime, c='red')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Best fitness", fontsize=18, c='black')
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(f"{f_name}.png")
        plt.savefig(f"{f_name}.pdf")
        print(f"{f_name} :  created")
    plt.show()


"""plotVelocities(velocities_iter, f"./plots/Ackley_{dims}D_velocity_0", save=True)
plotConvergence(global_best_per_iter, f"./plots/Ackley_{dims}D_convergence_0", save=True)"""

search_space = [[left_extreme, right_extreme]] * dims

# run vanilla
# save initial populations
populations = []
gbest_run_nostall = []
for r in range(30):
    FP = FuzzyPSO()  # StallFuzzyPSO()
    FP.set_fitness(ackley)
    FP.set_search_space(search_space)
    populations.append(copy.deepcopy([p.X for p in FP.Solutions]))
    global_best_per_iter = []
    FP.solve_with_fstpso(callback=callback_param)
    gbest_run_nostall.append(global_best_per_iter)

# run stall detection and perturbation
gbest_run_stall = []
for r in range(30):
    FP = StallFuzzyPSO()
    FP.set_fitness(ackley)
    FP.set_search_space(search_space)
    global_best_per_iter = []
    FP.solve_with_fstpso(callback=callback_param, initial_guess_list=populations[r])
    gbest_run_stall.append(global_best_per_iter)


