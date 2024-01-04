# every 10% maxiter, stalling particles' pbest +- 0.5

import copy
import logging
import math
import sys
import random

import numpy as np
from fstpso import *
from fstpso.fstpso_checkpoints import Checkpoint
from numpy import linalg, mean, zeros


# import fstpso


class StallParticle(fstpso.Particle):
    def __init__(self):
        fstpso.Particle.__init__(self)
        self.IsStalling = False
        self.Tau = 0
        self.Kappa = 0
        self.PreviousPbest = []


class FuzzyPSO_remedy_2(fstpso.FuzzyPSO):
    def __init__(self):
        fstpso.FuzzyPSO.__init__(self)
        self.CurrentCumAvg = 0
        self.AvgCounter = 0
        self.DetectStallInterval = 1
        self.SigmaPerc = 0.005
        self.KappaMax = 0
        self.DeltaVelocity = []
        self.SocialOverTime = []
        self.CognitiveOverTime = []
        self.InertiaOverTime = []

    def NewCreateParticles(self, n, dim, creation_method={'name': "uniform"}, initial_guess_list=None):

        del self.Solutions[:]

        for i in range(n):

            p = StallParticle()

            # dilation stuff
            p.ID = i

            p.X = self.NewGenerate([0] * dim, creation_method=creation_method)
            p.B = copy.deepcopy(p.X)
            p.V = list(zeros(dim))
            p.b0 = [0] * dim
            self.Solutions.append(p)

            if len(self.Solutions) == 1:
                self.G = copy.deepcopy(p)
                self.G.CalculatedFitness = sys.float_info.max
                self.W = copy.deepcopy(p)
                self.W.CalculatedFitness = sys.float_info.min

            # dilation stuff
            p.PreviousCenter = copy.deepcopy(p.X)
            p.PreviousX = list(zeros(dim))

        if initial_guess_list is not None:
            if len(initial_guess_list) > n:
                print(
                    "ERROR: the number of provided initial guesses (%d) is greater than the swarm size (%d), aborting." % (
                        len(initial_guess_list), n))
                exit(17)
            for i, ig in enumerate(initial_guess_list):
                if len(ig) != dim:
                    print("ERROR: each initial guess must have length equal to %d, aborting." % dim)
                    exit(18)
                else:
                    self.Solutions[i].X = copy.deepcopy(ig)
                    self.Solutions[i].B = copy.deepcopy(ig)
                    self.Solutions[i].CalculatedFitness = self.FITNESS(ig)

            print(" * %d particles created, initial guesses added to the swarm." % n)
        else:
            print(" * %d particles created." % n)

        print(" * FST-PSO will now assess the local and global best particles.")

        self.numberofparticles = n

        # if not self.ParallelFitness:
        self.UpdateCalculatedFitness()  # experimental

        vectorFirstFitnesses = [x.CalculatedFitness for x in self.Solutions]
        self.EstimatedWorstFitness = max(vectorFirstFitnesses)
        print(" * Estimated worst fitness: %.3f" % self.EstimatedWorstFitness)

        self.UpdateLocalBest(verbose=False, semiverbose=False)
        self.UpdatePositions()

        self.Dimensions = dim

        self.L = max([self.Boundaries[n][1] - self.Boundaries[n][0] for n in range(0, self.Dimensions)])
        self.diag = self.L * math.sqrt(dim)

        # self.DilationInfo['alpha'] = L

        # self.fattore3 = [0]*dim

        logging.info(' * %d particles created.' % (self.numberofparticles))

        self._used_generation_method = creation_method

    def _prepare_for_optimization(self, max_iter=None, max_iter_without_new_global_best=None, max_FEs=None,
                                  KappaMax=None, verbose=False):
        #  determine the Fitness Evalations budget (FEs)
        # 				- if the user specified a max_FEs, calculate the iterations according to the number of individuals.
        #                 Please note that the number of individuals may be not constant in the case of linearly decreasing populations.
        #               - if the user specified a max_iter, calculate the max_FEs according to the number of individuals.

        if max_iter is None and max_FEs is None:
            max_iter = 100  # default

        if max_FEs is None:
            max_FEs = self.numberofparticles * max_iter
        else:
            self._FES = max_FEs
            self.CheckStallInterval = 1
            max_iter = self._count_iterations(self._FES)

        self.MaxIterations = max_iter
        # self.RadiiOverTime = [[]]*max_iter

        if verbose:
            print(" * Iterations set to %d" % max_iter)
            print(" * Fitness evaluations budget set to %d" % max_FEs)

        if max_iter_without_new_global_best is not None:
            if max_iter < max_iter_without_new_global_best:
                print("WARNING: the maximum number of iterations (%d) is smaller than" % max_iter)
                print(
                    "         the maximum number of iterations without any update of the global best (%d)" % max_iter_without_new_global_best)
            self.MaxNoUpdateIterations = max_iter_without_new_global_best
            print(
                " * Maximum number of iterations without any update of the global best set to %d" % max_iter_without_new_global_best)
        self._overall_fitness_evaluations = 0

    def updateMean(self, CurrentCumAvg, newValue, currentSize):
        newSize = currentSize + 1
        updated = (currentSize * CurrentCumAvg + newValue) / (newSize)

        return updated, newSize

    def DetectStall(self):
        currentCumAvgVelocity = self.CurrentCumAvg
        numStallingParticles = 0
        for s in self.Solutions:
            # check velocity vs mean, last update in local best, ecc
            # update s.IsStalling
            particleVelocity = linalg.norm(s.V)
            if s.IsStalling:
                numStallingParticles += 1
            if particleVelocity <= currentCumAvgVelocity:
                s.Kappa += 1
                if s.Kappa <= self.KappaMax:
                    continue
                else:
                    s.IsStalling = True
                    s.Tau += 1
            else:
                s.IsStalling = False
                s.Kappa = 0
                s.Tau = 0

        return numStallingParticles

    def PerturbPBests(self):
        perturbedParticles = [False] * self.numberofparticles
        B = self.Boundaries
        D = len(B)
        for i in range(self.numberofparticles):
            s = self.Solutions[i]
            # if s has just been updated, reset counters --> no perturbation  # is the global best, or it
            if s.SinceLastLocalUpdate < 2:  # i == self.GIndex or
                s.Kappa = 0
                s.Tau = 0
                continue
            # perturb and reset counters
            perturbation = [np.random.normal(-0.5, 0.5, 1)[0] for _ in range(D)]
            s.B = [s.B[j] + perturbation[j] for j in range(D)]
            perturbedParticles[i] = True
        return perturbedParticles

    def UpdateVelocities(self, ParticlesToUpdate=None):
        """
            Update the velocity of all particles, according to their own settings.
        """

        if not ParticlesToUpdate:
            ParticlesToUpdate = [True] * self.numberofparticles

        for i in range(self.numberofparticles):
            p = self.Solutions[i]
            if ParticlesToUpdate[i]:
                for n in range(len(p.X)):

                    fattore1 = p.Inertia * p.V[n]
                    fattore2 = random.random() * p.CognitiveFactor * (p.B[n] - p.X[n])
                    fattore3 = random.random() * p.SocialFactor * (self.G.X[n] - p.X[n])

                    newvelocity = fattore1 + fattore2 + fattore3

                    # check max vel
                    if newvelocity > self.MaxVelocity[n] * p.MaxSpeedMultiplier:
                        newvelocity = self.MaxVelocity[n] * p.MaxSpeedMultiplier
                    elif newvelocity < -self.MaxVelocity[n] * p.MaxSpeedMultiplier:
                        newvelocity = -self.MaxVelocity[n] * p.MaxSpeedMultiplier

                    # check min vel
                    if abs(newvelocity) < self.MaxVelocity[n] * p.MinSpeedMultiplier:
                        newvelocity = math.copysign(self.MaxVelocity[n] * p.MinSpeedMultiplier, newvelocity)

                    # finally set velocity
                    p.V[n] = newvelocity  # * p.GammaInverter

    # logging.info('Particles velocities updated.')

    def Iterate(self, verbose=False):

        """self.SocialOverTime.append([p.SocialFactor for p in self.Solutions])
        self.CognitiveOverTime.append([p.CognitiveFactor for p in self.Solutions])
        self.InertiaOverTime.append([p.Inertia for p in self.Solutions])"""

        self.UpdateVelocities()

        # update current velocities
        currentVelocities = [linalg.norm(x.V) for x in self.Solutions]
        currentAvg = mean(currentVelocities)

        currentCumAvgVelocity = self.CurrentCumAvg
        currentSize = self.AvgCounter
        newCumAvgVelocity, newCounter = self.updateMean(currentCumAvgVelocity, currentAvg, currentSize)
        self.CurrentCumAvg = newCumAvgVelocity
        # update current velocities
        if self.Iterations and self.Iterations % self.DetectStallInterval == 0:
            self.DetectStall()

        perturbedParticles = self.PerturbPBests()

        velBefore = np.array([p.V for p in self.Solutions], dtype=object)[perturbedParticles]

        if np.any(perturbedParticles):
            for pp in range(len(perturbedParticles)):
                if perturbedParticles[pp]:
                    self.Solutions[pp].SocialFactor = 0
            # forzo aggiornamento della pbest fitness
            self.UpdateVelocities(perturbedParticles)


        velAfter = np.array([p.V for p in self.Solutions], dtype=object)[perturbedParticles]

        deltaV = np.asarray(velAfter-velBefore).astype(np.float64)
        self.DeltaVelocity.append(np.sqrt(np.einsum('ij,ij->i', deltaV, deltaV)))

        self.UpdatePositions()
        self.UpdateCalculatedFitness()
        self.UpdateLocalBest(verbose, semiverbose=False)
        if self._checkpoint is not None:
            S = Checkpoint(self)
            S.save_checkpoint(self._checkpoint, verbose)
            del S
        self.Iterations = self.Iterations + 1
        self.SinceLastGlobalUpdate = self.SinceLastGlobalUpdate + 1

    def UpdateLocalBest(self, verbose=False, semiverbose=True):

        if verbose:
            print("Beginning the verification of local bests")
        for i in range(len(self.Solutions)):
            if verbose:
                print(" Solution", i, ":", self.Solutions[i])

            self.Solutions[i].PreviousPbest = copy.deepcopy(self.Solutions[i].B)
            if self.Solutions[i].CalculatedFitness < self.Solutions[i].CalculatedBestFitness:
                self.Solutions[i].SinceLastLocalUpdate = 0
                if verbose: print(" * New best position for particle", i, "has fitness",
                                  self.Solutions[i].CalculatedFitness)

                self.Solutions[i].B = copy.deepcopy(self.Solutions[i].X)
                self.Solutions[i].CalculatedBestFitness = self.Solutions[i].CalculatedFitness
                if self.Solutions[i].CalculatedFitness < self.G.CalculatedFitness:
                    self.G = copy.deepcopy(self.Solutions[i])
                    if verbose or semiverbose:
                        print(" * New best particle in the swarm is #%d with fitness %f (it: %d)." % (
                        i, self.Solutions[i].CalculatedFitness, self.Iterations))

                    if self._discrete_cases is not None:
                        self._best_discrete_sample = self.Solutions[i]._last_discrete_sample

                    self.SinceLastGlobalUpdate = 0
                    self.GIndex = i
            else:
                if verbose: print(" Fitness calculated:", self.Solutions[i].CalculatedFitness, "old best",
                                  self.Solutions[i].CalculatedBestFitness)
                self.Solutions[i].SinceLastLocalUpdate += 1
                if self.G.X != self.Solutions[i].B:
                    if self.Solutions[i].SinceLastLocalUpdate > self._threshold_local_update:
                        self.Solutions[i]._mark_for_restart()
                        if verbose: print(" * Particle %d marked for restart" % i)

                # update global worst
                if self.Solutions[i].CalculatedFitness > self.W.CalculatedFitness:
                    self.W = copy.deepcopy(self.Solutions[i])
                    self.WIndex = i

        if self.Iterations > 0:
            logging.info('[Iteration %d] best individual fitness: %f' % (self.Iterations, self.G.CalculatedFitness))
            logging.info('[Iteration %d] best individual structure: %s' % (self.Iterations, str(self.G.X)))

    def TerminationCriterion(self, verbose=False):

        if verbose:
            print("Iteration:", self.Iterations)
            print(", since last global update:", self.SinceLastGlobalUpdate)

        if self.StopOnGoodFitness == True:
            if self.G.CalculatedFitness < self.GoodFitness:
                if verbose:
                    print("Good fitness reached!", self.G.CalculatedFitness)
                return True

        if self.SinceLastGlobalUpdate > self.MaxNoUpdateIterations:
            if verbose:
                print("Too many iterations without new global best")
            return True

        if self.Iterations >= self.MaxIterations:
            if verbose:
                print("Maximum iterations reached")
            return True
        else:
            return False

    def solve_with_fstpso(self,
                          max_iter=None, max_iter_without_new_global_best=None, max_FEs=None,
                          creation_method={'name': "uniform"},
                          initial_guess_list=None,
                          save_checkpoint=None,
                          restart_from_checkpoint=None,
                          callback=None, verbose=False,
                          KappaMax=None,
                          dump_best_fitness=None, dump_best_solution=None):
        """
            Launches the optimization using FST-PSO. Internally, this method checks
            that we correctly set the pointer to the fitness function and the
            boundaries of the search space.

            Args:
                max_iter: the maximum number of iterations of FST-PSO
                creation_method: specifies the type of particles initialization
                initial_guess_list: the user can specify a list of initial guesses for particles
                                    to accelerate the convergence (BETA)
                dump_best_fitness: at the end of each iteration fst-pso will save to this file the best fitness value
                dump_best_solution: at the end of each iteration fst-pso will save to this file the structure of the
                                    best solution found so far
                callback: this argument accepts a dictionary with two items: 'function' and 'interval'. Every
                          'interval' iterations, the 'function' is called. This functionality can be exploited
                          to implement special behavior (e.g., restart)
                save_checkpoint: save a checkpoint to the specified path in order to recover from crashes
                restart_from_checkpoint: restart the optimization from the specified checkpoint
                verbose: enable verbose mode

            Returns:
                This method returns a couple (optimal solution, fitness of the optimal solution)
                :param KappaMax:
        """

        self.DetectStallInterval = 0.1 * self.MaxIterations
        self.KappaMax = int(math.log(self.MaxIterations))
        self.NewCreateParticles(self.NumberOfParticles, self.Dimensions, initial_guess_list=initial_guess_list)

        # first step: check potential errors in FST-PSO's initialization
        self._check_errors()
        self._prepare_for_optimization(max_iter, max_iter_without_new_global_best, max_FEs, KappaMax, verbose)

        self.UseRestart = "reboot" in self.enabled_settings
        if self.UseRestart:
            self._threshold_local_update = max(30, int(max_iter / 20))
            if verbose: print(" * Reboots are activated with theta=%d" % self._threshold_local_update)

        if save_checkpoint is not None:
            self._checkpoint = save_checkpoint

        if restart_from_checkpoint is None:
            if verbose:
                print(" * Creating and evaluating random particles")
            self.NewCreateParticles(self.numberofparticles, self.dimensions, creation_method=creation_method,
                                    initial_guess_list=initial_guess_list)
            self._overall_fitness_evaluations += self.numberofparticles
            self.Iterations = 0
            if initial_guess_list is not None:
                self._overall_fitness_evaluations -= self.numberofparticles
        else:
            self._load_checkpoint(restart_from_checkpoint, verbose)
        self.MaxIterations = int((max_FEs - self.numberofparticles) / self.numberofparticles)
        return self._actually_launch_optimization(verbose=verbose, callback=callback,
                                                  dump_best_solution=dump_best_solution,
                                                  dump_best_fitness=dump_best_fitness)
