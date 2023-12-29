from unittest import skip
from miniful import *
import math
import logging
from numpy import random, array, linalg, zeros, argmin, argsort, exp, mean, sign
from numpy.random import choice
import os
import copy
import random
from numpy.random import lognormal, uniform
import sys
import subprocess
import pickle
from copy import deepcopy
from .fstpso_checkpoints import Checkpoint
from itertools import chain
import matplotlib.pyplot as plt
from . import df
from scipy.stats import linregress


class Particle(object):

    def __init__(self):
        self.X = []
        self.V = []
        self.B = []
        self.B_discrete = None
        self.MarkedForRestart = False
        self.CalculatedFitness = sys.float_info.max
        self.FitnessDevStandard = sys.float_info.max
        self.CalculatedBestFitness = sys.float_info.max
        self.SinceLastLocalUpdate = 0

        self.DerivativeFitness = 0
        self.MagnitudeMovement = 0
        self.DistanceFromBest = sys.float_info.max
        self.CognitiveFactor = 2.
        self.SocialFactor = 2.
        self.Inertia = 0.5

        # support for PPSO
        self.MaxSpeedMultiplier = .25
        self.MinSpeedMultiplier = 0
        self.GammaInverter = 1

        self.cannot_move = False

        # used in the case of discrete optimization
        self._last_discrete_sample = None

        # Dilation stuff
        self.IsStalling = False
        # self.StallingPeriods = []
        self.ID = -1
        self.IterationsStalling = 0
        self.VelocityOverTime = [0]
        # self.RadiusOverTime = []
        self.R = 0
        self.alpha = 1
        self.gamma = 0
        self.beta = 1
        self.PreviousR = 0
        self.PreviousAlpha = 0
        self.PreviousCenter = []
        self.PreviousGamma = 1
        self.PreviousBeta = 0
        self.PreviouslyStalling = False
        self.SinceVelocityLessThanAvg = 0
        # self.IterationsStalling = 0
        self.PreviousX = []
        self.DilationEffect = 0
        """self.counter_switches = 0
		self.counter_negatives = 0
		self.counter_positives = 0
		self.counter_zeros = 0"""
        self.BubbleExceeded = 0

    def can_move(self):
        if self.cannot_move:
            self.cannot_move = False
            return False
        else:
            return True

    def __repr__(self):
        return "<Particle %s>" % str(self.X)

    def __str__(self):
        return "\t".join(map(str, self.X))

    def _mark_for_restart(self):
        self.MarkedForRestart = True


class PSO_new(object):

    def __repr__(self):
        return str("<PSO instance " + self.ID + ">")

    def __init__(self, ID=""):
        self.G = None
        self.W = None  # worst

        self.ID = ID

        self.GIndex = 0
        self.WIndex = 0

        self.Solutions = []
        self.InertiaStart = 0.9
        self.InertiaEnd = 0.4
        self.Inertia = 0.9
        self.CognitiveFactor = 1.9
        self.SocialFactor = 1.9
        self.MaxVelocity = 1.0
        self.UseRestart = False
        self.UseLog = False
        self.ProximityThreshold = 1E-1

        if sys.version_info[0] < 3:
            maximum_integer = sys.maxint
        else:
            maximum_integer = sys.maxsize

        self.MaxIterations = maximum_integer
        self.MaxNoUpdateIterations = maximum_integer
        self.GoodFitness = -sys.float_info.max
        self.Iterations = 0
        self.SinceLastGlobalUpdate = 0
        self.FITNESS = None
        self.Boundaries = []
        self.save_path = ""
        self.NumberOfParticles = 0
        self.Color = "black"
        self.Nickname = "Standard"
        self.StopOnGoodFitness = False
        self.EstimatedWorstFitness = sys.float_info.max
        self.Dimensions = 0
        self._used_generation_method = None
        self._threshold_local_update = 50

        self._checkpoint = None

        # for discrete optimization
        self._best_discrete_sample = None

        # self.SIMULATOR_PATH = None
        self.ARGUMENTS = None
        self.COMMUNICATION_FILE = None

        # change this for parallel fitness calculation
        self.ParallelFitness = False

        # Dilation stuff
        self.VelocitiesOverTime = []
        # self.MeanOverTime = []
        self.CurrentCumAvg = 0
        self.AvgCounter = 0
        self.UseDilation = False
        # self.DilationInfo = None
        self.CheckStallInterval = 1  # 25
        # self.PersonalBestInterval = self._threshold_local_update / 2
        self.MaxIterationsWIthDilation = 0
        # self.StallOverTime = []
        # self.RadiiOverTime = []
        # self.Destalled = 0
        self.Radii = []
        self.Seed = -1
        self.SeedNumber = -1
        # self.MaxSparseness = 1
        # self.Bouncing = 0
        # self.Pacman = 0
        # self.NUpdates = 0
        self.IterationsWithVelocityLessThanAvg = 0
        self.funname = ""
        self.SignProbabilities = [0.5, 0.5]
        # self.MaxVelocityNorm = 1

        # self.SparsenessOverTime = []
        # self.NormalizationDistance = 1

        self._print_banner()

    def CalculateRadius(self, n, x):
        L = self.Boundaries[n][1] - self.Boundaries[n][0]
        return L * math.floor(x) / self._threshold_local_update

    def CompareVelocities(self):
        currentCumAvgVelocity = self.CurrentCumAvg
        # self.MeanOverTime.append(currentCumAvgVelocity)
        stallCounter = 0
        for s in self.Solutions:
            # check velocity vs mean, last update in local best, ecc
            # update s.IsStalling
            particleVelocity = linalg.norm(s.V)
            if s.IsStalling:
                stallCounter += 1
            # no updates in personal best for #PersonalBestInterval iterations & velocity < average
            # and s.SinceLastLocalUpdate > self.PersonalBestInterval
            # if particleVelocity:# <= self.MaxVelocityNorm:
            if particleVelocity <= currentCumAvgVelocity:
                s.SinceVelocityLessThanAvg += 1
                if s.SinceVelocityLessThanAvg <= self.IterationsWithVelocityLessThanAvg:
                    continue
                else:
                    s.IsStalling = True
                    s.IterationsStalling += 1
                # s.SocialFactorEnabled = 0
            else:
                s.IsStalling = False
                s.SinceVelocityLessThanAvg = 0
                s.IterationsStalling = 0
                # s.SocialFactorEnabled = 1
                """if s.SinceLastLocalUpdate > self.PersonalBestInterval:
				#if True:
					s.IsStalling = True
					s.StallingPeriods.append([self.Iterations])
					stallCounter += 1
				continue"""
            # and particleVelocity >= currentCumAvgVelocity
            """if s.IsStalling and particleVelocity > currentCumAvgVelocity: #and s.IterationsWithDilation > self.MaxIterationsWithDilation:
				if s.SinceLastLocalUpdate <= self.PersonalBestInterval:
					continue
				s.IsStalling = False
				s.IterationsWithDilation = 0
				self.Destalled += 1
				s.StallingPeriods[-1].append(self.Iterations)"""
        #
        return stallCounter
        """for s in self.Solutions:
			s.IsStalling = True
		return len(self.Solutions)"""

    def _print_banner(self):
        import pkg_resources
        vrs = pkg_resources.get_distribution('fst-pso').version
        print("Fuzzy Self-Tuning PSO - v%s" % vrs)

    def getBestIndex(self):
        index = 0
        best_fitness = self.Solutions[0].CalculatedFitness
        for n, s in enumerate(self.Solutions):
            if s.CalculatedFitness < best_fitness:
                best_fitness = s.CalculatedFitness
                index = n
        return index

    def getWorstIndex(self):
        index = 0
        worst_fitness = self.Solutions[0].CalculatedFitness
        for n, s in enumerate(self.Solutions):
            if s.CalculatedFitness > worst_fitness:
                worst_fitness = s.CalculatedFitness
                index = n
        return index

    def UpdateInertia(self):
        self.Inertia = self.InertiaStart - (self.InertiaStart - self.InertiaEnd) / self.MaxIterations * self.Iterations

    def Iterate(self, verbose=False):
        self.UpdateVelocities()

        # update currentvelocities
        currentVelocities = [linalg.norm(x.V) for x in self.Solutions]
        currentAvg = mean(currentVelocities)
        self.VelocitiesOverTime.append(currentVelocities)

        currentCumAvgVelocity = self.CurrentCumAvg
        currentSize = self.AvgCounter
        newCumAvgVelocity, newCounter = self.updateMean(currentCumAvgVelocity, currentAvg, currentSize)
        self.CurrentCumAvg = newCumAvgVelocity
        self.AvgCounter = newCounter
        # update currentvelocities

        if self.Iterations and self.Iterations % self.CheckStallInterval == 0:
            self.CompareVelocities()

        self.UpdatePositions()
        self.UpdateCalculatedFitness()
        self.UpdateLocalBest(verbose, semiverbose=False)
        if self._checkpoint is not None:
            S = Checkpoint(self)
            S.save_checkpoint(self._checkpoint, verbose)
            del S
        # En = self.CalculateSparseness()
        # self.SparsenessOverTime.append(En)
        self.Iterations = self.Iterations + 1
        self.SinceLastGlobalUpdate = self.SinceLastGlobalUpdate + 1

    # self.SignProbabilities[0] += 0.5/self.MaxIterations
    # self.SignProbabilities[1] -= 0.5/self.MaxIterations

    def updateMean(self, CurrentCumAvg, newValue, currentSize):
        newSize = currentSize + 1
        updated = (currentSize * CurrentCumAvg + newValue) / (newSize)

        return updated, newSize

    def Solve(self, funz, verbose=False, callback=None, dump_best_fitness=None, dump_best_solution=None,
              print_bar=True):

        logging.info('Launching optimization.')

        if verbose:
            print(f" * Process started, funname: {self.funname}, seedn: {self.SeedNumber}")

        i = 0

        while (not self.TerminationCriterion(verbose=verbose)):
            if funz != None:    funz(self)

            self.Iterate(verbose)
            if verbose:
                print("Completed iteration %d" % (self.Iterations))
            else:
                if print_bar:
                    if self.Iterations in [self.MaxIterations // x for x in range(1, 10)]:
                        print(" * %dth iteration out of %d completed. " % (self.Iterations, self.MaxIterations), end="")
                        print("[%s%s]" %
                              (
                                  "#" * int(30 * self.Iterations / self.MaxIterations),
                                  " " * (30 - int(30 * self.Iterations / self.MaxIterations))
                              )
                              )

            # new: if a callback is specified, call it at regular intervals
            if callback != None:
                interval = callback['interval']
                function = callback['function']
                if (self.Iterations - 1) % interval == 0:
                    function(self)
                    if verbose: print(" * Callback invoked")

            # write the current best fitness 
            if dump_best_fitness != None:
                if self.Iterations == 1:
                    with open(dump_best_fitness, "w") as fo: pass  # touch
                with open(dump_best_fitness, "a") as fo:
                    fo.write(str(self.G.CalculatedFitness) + "\n")

            # write the current best solution
            if dump_best_solution != None:
                if self.Iterations == 1:
                    with open(dump_best_solution, "w") as fo: pass  # touch
                with open(dump_best_solution, "a") as fo:
                    fo.write("\t".join(map(str, self.G.X)) + "\n")

        if verbose:
            print(" * Optimization process terminated. Best solution found:", self.G.X, "with fitness",
                  self.G.CalculatedFitness)

        logging.info('Best solution: ' + str(self.G))
        logging.info('Fitness of best solution: ' + str(self.G.CalculatedFitness))

        # print (self._discrete_cases)
        if self._discrete_cases is None:
            return self.G, self.G.CalculatedFitness
        else:
            return self.G, self.G.CalculatedFitness, self._best_discrete_sample

    def TerminationCriterion(self, verbose=False):

        if verbose:
            print("Iteration:", self.Iterations),
            print(", since last global update:", self.SinceLastGlobalUpdate)

        if self.StopOnGoodFitness == True:
            if self.G.CalculatedFitness < self.GoodFitness:
                if verbose:
                    print(" * Optimal fitness was reached", self.G.CalculatedFitness)
                return True

        if self.SinceLastGlobalUpdate > self.MaxNoUpdateIterations:
            if verbose:
                print(" * Too many iterations without new global best")
            return True

        if self.Iterations >= self.MaxIterations:
            if verbose:
                print(" * Maximum iterations reached")
            return True
        else:
            return False

    def NewGenerate(self, lista, creation_method):

        ret = []

        if creation_method['name'] == "uniform":
            for i in range(len(lista)):
                ret.append(self.Boundaries[i][0] + (self.Boundaries[i][1] - self.Boundaries[i][0]) * random.random())

        elif creation_method['name'] == "logarithmic":

            for i in range(len(lista)):
                if self.Boundaries[i][0] < 0:
                    minimo = -5
                    massimo = math.log(self.Boundaries[i][1])
                    res = math.exp(minimo + (massimo - minimo) * random.random())
                    if random.random() > .5:
                        res *= -1
                    ret.append(res)
                else:
                    minimo = math.log(self.Boundaries[i][0])
                    massimo = math.log(self.Boundaries[i][1])
                    ret.append(math.exp(minimo + (massimo - minimo) * random.random()))

        elif creation_method['name'] == "normal":
            for i in range(len(lista)):
                while (True):
                    if self.Boundaries[i][1] == self.Boundaries[i][0]:
                        ret.append(self.Boundaries[i][1])
                        break
                    cand_position = random.gauss((self.Boundaries[i][1] + self.Boundaries[i][0]) / 2,
                                                 creation_method['sigma'])
                    if (cand_position >= self.Boundaries[i][0] and cand_position <= self.Boundaries[i][1]):
                        ret.append(cand_position)
                        break

        elif creation_method['name'] == "lognormal":
            for i in range(len(lista)):
                minord = math.log(self.Boundaries[i][0])
                maxord = math.log(self.Boundaries[i][1])
                media = (maxord + minord) / 2.
                while (True):
                    if self.Boundaries[i][1] == self.Boundaries[i][0]:
                        ret.append(self.Boundaries[i][1])
                        break
                    v = lognormal(media, creation_method['sigma'])
                    if v >= self.Boundaries[i][0] and v <= self.Boundaries[i][1]:
                        break
                ret.append(v)

        else:
            raise Exception("Unknown particles initialization mode")

        return ret

    def NewCreateParticles(self, n, dim, creation_method={'name': "uniform"}, initial_guess_list=None):

        del self.Solutions[:]

        for i in range(n):

            p = Particle()

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

    # conventional PSO 
    def UpdateCalculatedFitness(self):
        for s in self.Solutions:
            prev = s.CalculatedFitness
            ret = self.FITNESS(s.X)
            if s.MagnitudeMovement != 0:
                s.DerivativeFitness = (ret - prev) / s.MagnitudeMovement
            if isinstance(ret, list):
                s.CalculatedFitness = ret[0]
                s.Differential = ret[1]
            else:
                s.CalculatedFitness = ret

    def UpdateLocalBest(self, verbose=False, semiverbose=True):

        if verbose:
            print("Beginning the verification of local bests")
        for i in range(len(self.Solutions)):
            if verbose:
                print(" Solution", i, ":", self.Solutions[i])
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


class FuzzyPSO(PSO_new):

    def __init__(self, logfile=None):
        """
		Creates a new FST-PSO instance.

		Args:
			D: number of dimensions of the problem under optimization
		"""

        super(FuzzyPSO, self).__init__()

        # defaults for membership functions
        self.DER1 = -1.0
        self.DER2 = 1.0

        self.MDP1 = 0.2
        self.MDP2 = 0.4
        self.MDP3 = 0.6

        self.MaxDistance = 0
        self.dimensions = 0
        self._overall_fitness_evaluations = 0
        self._FES = None

        self.enabled_settings = ["cognitive", "social", "inertia", "minvelocity", "maxvelocity"]

        self._FITNESS_ARGS = None

        self.LOW_INERTIA = 0.3
        self.MEDIUM_INERTIA = 0.5
        self.HIGH_INERTIA = 1.0

        self.LOW_SOCIAL = 1.0
        self.MEDIUM_SOCIAL = 2.0
        self.HIGH_SOCIAL = 3.0

        self.LOW_COGNITIVE = 0.1
        self.MEDIUM_COGNITIVE = 1.5
        self.HIGH_COGNITIVE = 3.0

        self.LOW_MINSP = 0
        self.MEDIUM_MINSP = 0.001
        self.HIGH_MINSP = 0.01

        self.LOW_MAXSP = 0.1
        self.MEDIUM_MAXSP = 0.15
        self.HIGH_MAXSP = 0.2

        self._discrete_cases = None

        self._norm_version = "FST-PSO2b"

        """ 
		self.LOW_GAMMA = -1
		self.MEDIUM_GAMMA = 0.9
		self.HIGH_GAMMA = 1
		"""

        if logfile != None:
            print(" * Logging to file", logfile, "enabled")
            with open(logfile, 'w'): pass
            logging.basicConfig(filename=logfile, level=logging.DEBUG)
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
            logging.info('FST-PSO object created.')

    def enable_decreasing_population(self, FES):
        print(
            " * Fitness evaluations budget was specified, the 'max_iter' argument will be discarded and the swarm size will be ignored.")
        self.enabled_settings.append("lin_pop_decrease")

    def enable_reboot(self):
        self.enabled_settings += ["reboot"]
        print(" * Reboots ENABLED")

    def disable_fuzzyrule_cognitive(self):
        self.enabled_settings = filter(lambda x: x != "cognitive", self.enabled_settings)
        print(" * Fuzzy rules for cognitive factor DISABLED")

    def disable_fuzzyrule_social(self):
        self.enabled_settings = filter(lambda x: x != "social", self.enabled_settings)
        print(" * Fuzzy rules for social factor DISABLED")

    def disable_fuzzyrule_inertia(self):
        self.enabled_settings = filter(lambda x: x != "inertia", self.enabled_settings)
        print(" * Fuzzy rules for inertia weight DISABLED")

    def disable_fuzzyrule_maxvelocity(self):
        self.enabled_settings = filter(lambda x: x != "maxvelocity", self.enabled_settings)
        print(" * Fuzzy rules for maximum velocity cap DISABLED")

    def disable_fuzzyrule_minvelocity(self):
        self.enabled_settings = filter(lambda x: x != "minvelocity", self.enabled_settings)
        print(" * Fuzzy rules for minimum velocity cap DISABLED")

    def _count_iterations(self, FES):
        NFEcur = 0
        curpop = 0
        SEQ = []

        if "lin_pop_decrease" in self.enabled_settings:
            while (NFEcur < FES):
                curpop = self._get_pop_size(NFEcur)
                NFEcur += curpop
                SEQ.append(curpop)
        else:
            print(" * Determining the number of iterations given the FE budget (%d)" % FES)
            curpop = self.numberofparticles
            NFEcur = curpop
            SEQ.append(curpop)
            while (1):
                if NFEcur + curpop > FES:
                    curpop = FES - NFEcur
                SEQ.append(curpop)
                NFEcur += curpop
                if NFEcur >= FES:
                    break

        # print(SEQ, sum(SEQ)); exit()

        est_iterations = len(SEQ) - 1

        return est_iterations

    def _check_errors(self):

        if self.FITNESS is None:
            print("ERROR: cannot solve a problem without a fitness function; use set_fitness()")
            exit(-3)

        if self.Boundaries == []:
            print("ERROR: FST-PSO cannot solve unbounded problems; use set_search_space()")
            exit(-4)

    def _prepare_for_optimization(self, max_iter=None, max_iter_without_new_global_best=None, max_FEs=None,
                                  verbose=False):
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
            # tmp = int(0.0005 * max_FEs) #0.0005 --> 0.00015
            self.IterationsWithVelocityLessThanAvg = math.log(max_FEs)  # tmp #math.log(max_FEs)#tmp
            self.CheckStallInterval = 1  # tmp
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

    def solve_with_fstpso(self,
                          max_iter=None, max_iter_without_new_global_best=None, max_FEs=None,
                          creation_method={'name': "uniform"},
                          initial_guess_list=None,
                          save_checkpoint=None,
                          restart_from_checkpoint=None,
                          callback=None, verbose=False,
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
		"""

        # first step: check potential errors in FST-PSO's initialization
        self._check_errors()
        self._prepare_for_optimization(max_iter, max_iter_without_new_global_best, max_FEs, verbose)

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

            """# save initial population
			with open(f"first_population_d_{self.Dimensions}.ser","wb") as f:
				population = [s.X for s in self.Solutions]
				pickle.dump(population,f)"""

            self._overall_fitness_evaluations += self.numberofparticles
            self.Iterations = 0
        else:
            self._load_checkpoint(restart_from_checkpoint, verbose)

        # Dilation stuff
        # self.MaxSparseness = self.CalculateMaxSparseness()
        # self.NormalizationDistance = 2*pow(self.numberofparticles,2)*math.sqrt(self.Dimensions)*max([self.Boundaries[n][1] - self.Boundaries[n][0] for n in range(0,self.Dimensions)])

        return self._actually_launch_optimization(verbose=verbose, callback=callback,
                                                  dump_best_solution=dump_best_solution,
                                                  dump_best_fitness=dump_best_fitness)

    def _actually_launch_optimization(self, verbose=None, callback=None, dump_best_solution=None,
                                      dump_best_fitness=None):
        if verbose:
            print(" * Enabled settings:", " ".join(map(lambda x: "[%s]" % x, self.enabled_settings)))
        print(f"\n *** All prepared, launching optimization, funname: {self.funname}, seedn: {self.SeedNumber}***")
        result = self.Solve(None, verbose=verbose, callback=callback, dump_best_solution=dump_best_solution,
                            dump_best_fitness=dump_best_fitness)
        return result

    def _load_checkpoint(self, restart_from_checkpoint, verbose=False):
        if verbose: print(" * Restarting the optimization from checkpoint '%s'..." % restart_from_checkpoint)
        g = open(restart_from_checkpoint, "rb")
        obj = pickle.load(g)
        self.Solutions = deepcopy(obj._Solutions)
        self.G = deepcopy(obj._G)
        self.W = deepcopy(obj._W)
        self.Iterations = obj._Iteration

    def set_search_space(self, limits):
        """
			Sets the boundaries of the search space.

			Args:
			limits: it can be either a 2D list or 2D array, shape = (dimensions, 2).
					For instance, if you have D=3 variables in the real interval [-1,1]
					you can provide the boundaries as: [[-1,1], [-1,1], [-1,1]].
					The dimensions of the problem are automatically determined 
					according to the length of 'limits'.
		"""
        D = len(limits)
        self.dimensions = D

        self.MaxDistance = max_distance = calculate_max_distance(limits)
        print(" * Max distance: %f" % self.MaxDistance)

        self.Boundaries = limits
        print(" * Search space boundaries set to:", limits)

        self._set_maxvelocity()
        self._set_num_particles()

        logging.info('Search space set (%d dimensions).' % (D))

    def set_search_space_discrete(self, limits):
        """
			Sets the boundaries of the search space.

			Args:
			limits: it can be either a list or an array, shape = (dimensions, choices).
					For instance, if you have D=3 variables such that: the first one
					can take the discrete values [1,2]; the second one can be [0,2,4];
					the third one can be [10,100]. Then you can provide the boundaries as: 
					[[1,2], [0,2,4], [10,100]].
					The dimensions of the problem are automatically determined 
					according to the length of 'limits'.
		"""

        # convert discrete interval into probability distributions
        D = sum([len(x) for x in limits])
        self._discrete_cases = limits[:]
        print(" * Discrete case detected, with the following choices:", self._discrete_cases)
        limits = [[0, 1]] * D
        self.dimensions = D

        self.MaxDistance = max_distance = calculate_max_distance(limits)
        print(" * Max distance: %f" % self.MaxDistance)

        self.Boundaries = limits
        print(" * FST-PSO converted the %dD discrete problem to a %dD real valued probabilistic problem." % (
        len(self._discrete_cases), D))
        print(" * Search space boundaries automatically set to:", limits)

        self._set_maxvelocity()
        self._set_num_particles()

        logging.info('Search space set (%d dimensions).' % (D))

    def _set_maxvelocity(self):
        self.MaxVelocity = [math.fabs(B[1] - B[0]) for B in self.Boundaries]
        # self.MaxVelocityNorm = linalg.norm(array(self.MaxVelocity), ord=2) / 100
        print(" * Max velocities set to:", self.MaxVelocity)

    def _set_num_particles(self):
        self.numberofparticles = int(10 + 2 * math.sqrt(self.dimensions))
        print(" * Number of particles automatically set to", self.numberofparticles)

    def set_swarm_size(self, N):
        """
			This (optional) method overrides FST-PSO's heuristic and 
			forces the number of particles in the swarm.alpha

		"""
        try:
            N = int(N)
        except:
            print("ERROR: please specify the swarm size as an integer number")
            exit(-6)

        if N <= 1:
            print("ERROR: FST-PSO cannot work with less than 1 particles, aborting")
            exit(-5)
        else:
            self.numberofparticles = N
            print(" * Swarm size now set to %d particles" % (self.numberofparticles))

        logging.info('Swarm size set to %d particles.' % (self.numberofparticles))

    def _dilate(self, X, **args):
        if args['method'] == "sigmoid":
            z = 1 / (1 + exp(-X))
            # print(list(zip(X,z)))
            return z
        elif args['method'] == 'smoothramp':
            alpha = args['alpha']
            z = 1 - 1 / (1 + (X * 2) ** alpha)
            # print(list(zip(X,z)))
            return z
        else:
            raise Exception("Dilation method unknown (%s)" % method)

    def _convert_prob_to_particle(self, data, use_dilation=False):
        """
			Generates a valid solution for the discrete optimization problem
			using the probability distributions encoded by data.
		"""
        original_D = len(self._discrete_cases)
        loc = 0
        sample = []
        for d in range(original_D):
            cases = len(self._discrete_cases[d])
            pseudoprob = array(data[loc:loc + cases])

            # check for probabilities all equal to zero
            if pseudoprob.sum() == 0:
                print("WARNING: probabilities are all zero")
                distribution = [1 / D for _ in range(D)]
            else:
                # normalize probabilities
                distribution = pseudoprob / pseudoprob.sum()

            # make distribution more extreme
            if use_dilation:
                distribution = self._dilate(distribution, method="smoothramp", alpha=8)

            sample.append(choice(self._discrete_cases[d], p=distribution))
            loc += cases
        return sample

    def call_fitness(self, particle, arguments=None):

        if self.FITNESS == None: raise Exception("ERROR: fitness function not valid")

        data = particle.X

        # in the discrete case, use the particle as probability distribution
        # and generate a new individual according to that. Store the generated structure
        # in order to return it later if the fitness is optimal.
        if self._discrete_cases is not None:
            data = self._convert_prob_to_particle(data)
            particle._last_discrete_sample = data

        if arguments == None:
            return self.FITNESS(data)
        else:
            return self.FITNESS(data, arguments)

    def set_fitness(self, fitness, arguments=None, skip_test=True):
        """
			Sets the fitness function used to evaluate the particles.
			This method performs an automatic validation of the fitness function
			(you can disable this feature by setting the optional argument 'skip_test' to True).

			Args:
			fitness : a user-defined function. The function must accept
			a vector of real-values as input (i.e., a candidate solution)
			and must return a real value (i.e., the corresponding fitness value)
			arguments : a dictionary containing the arguments to be passed to the fitness function.
			skip_test : True/False, bypasses the automatic fitness check.
	
		"""
        if skip_test:
            self.FITNESS = fitness
            self._FITNESS_ARGS = arguments
            self.ParallelFitness = False
            return

        self.FITNESS = fitness
        self._FITNESS_ARGS = arguments

        # there are two cases:  the user already specified the search space or not.
        # 						In the first case, FST-PSO will generate a random particle in the search space.
        #						In the second case, FST-PSO will test [[1e-10]]*D
        if self.Boundaries == []:
            test_particle = [[1e-10]] * self.dimensions
        else:
            test_particle = [uniform(x, y) for x, y in self.Boundaries]
        print(" * Testing fitness evaluation... ", end="")
        self.call_fitness(test_particle, self._FITNESS_ARGS)
        self.ParallelFitness = False
        print("test successful.")

    def set_parallel_fitness(self, fitness, arguments=None, skip_test=True):
        print(" * Parallel fitness requested")
        if skip_test:
            self.FITNESS = fitness
            self._FITNESS_ARGS = arguments
            self.ParallelFitness = True
            return

        np = Particle()
        np.X = [0] * self.dimensions
        self.FITNESS = fitness
        self._FITNESS_ARGS = arguments
        try:
            self.call_fitness([np.X] * self.numberofparticles, arguments)
        except:
            print("ERROR: the specified function does not seem to implement a correct fitness function")
            exit(-2)
        self.ParallelFitness = True
        print(" * Test successful")

    def phi(self, f_w, f_o, f_n, move, move_max):
        """ 
			Calculates the Fitness Incremental Factor (phi).

			Arguments: 
				f_w = estimated worst fitness
				f_o = previous fitness
				f_n = new fitness
				move = magnitude movement
				move_max = maximum distance
		"""
        if move == 0: return 0  # we did not move

        left = move / move_max
        if self._norm_version == "FST-PSO1":
            # right = (min(f_w, f_n) - min(f_w, f_o))/f_w
            right = (min(f_w, f_n) - min(f_w, f_o)) / abs(f_w)
        elif self._norm_version == "FST-PSO2a":
            if self.G.CalculatedFitness > 0:
                right = (min(f_w, f_n) - min(f_w, f_o)) / abs(f_w)
            else:
                right = (min(f_w, f_n) - min(f_w, f_o)) / abs(f_w - self.G.CalculatedFitness)
        elif self._norm_version == "FST-PSO2b":
            right = (min(f_w, f_n) - min(f_w, f_o)) / abs(f_w - self.G.CalculatedFitness)
        else:
            raise NotImplementedError()

        return left * right

    def CreateRadiusFuzzyReasoner(self, L, B, low_value=0):

        FR = FuzzyReasoner()
        L1 = L * math.sqrt(self.Dimensions)
        # L2 = math.sqrt(L*self.Dimensions)
        LOW_RADIUS = low_value
        MEDIUM_RADIUS = L1 / 2  # L2 #L1/2
        HIGH_RADIUS = L1

        HIGH_ALPHA = 10
        MEDIUM_ALPHA = 5
        LOW_ALPHA = 1

        HIGH_GAMMA = 1
        MEDIUM_GAMMA = 0.5
        LOW_GAMMA = 0

        HIGH_BETA = 10
        MEDIUM_BETA = 5
        LOW_BETA = 1

        USE_HQ = True

        """myFS1 = FuzzySet(points=[[0,1], [1.,0]],			term="CLOSE", high_quality_interpolate=USE_HQ)
		myFS2 = FuzzySet(points=[[0.5,1.],[0,0],[1.,0]],	term="MEDIUM", high_quality_interpolate=USE_HQ)
		myFS3 = FuzzySet(points=[[0,0], [1.,1.]],			term="FAR", high_quality_interpolate=USE_HQ)
		DISTANCE_MF = MembershipFunction( [myFS1,myFS2,myFS3], concept="DISTANCE")

		myFS4 = FuzzySet(points=[[0,0], [B,1.]],		term="HIGH", high_quality_interpolate=USE_HQ)
		myFS5 = FuzzySet(points=[[B/2,1.],[0,0],[B,0]],	term="MEDIUM", high_quality_interpolate=USE_HQ)
		myFS6 = FuzzySet(points=[[0,1.],[B,0]],			term="POOR", high_quality_interpolate=USE_HQ)
		BUDGET_MF = MembershipFunction( [myFS4,myFS5,myFS6], concept="BUDGET")"""

        myFS1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="CLOSE", high_quality_interpolate=USE_HQ)
        myFS2 = FuzzySet(points=[[0.5, 1.], [0, 0], [1., 0]], term="MEDIUM", high_quality_interpolate=USE_HQ)
        myFS3 = FuzzySet(points=[[0.5, 0], [1., 1.]], term="FAR", high_quality_interpolate=USE_HQ)
        DISTANCE_MF = MembershipFunction([myFS1, myFS2, myFS3], concept="DISTANCE")

        """myFS4 = FuzzySet(points=[[B/2,0], [B,1.]],		term="HIGH", high_quality_interpolate=USE_HQ)
		myFS5 = FuzzySet(points=[[B/2,1.],[0,0],[B,0]],	term="MEDIUM", high_quality_interpolate=USE_HQ)
		myFS6 = FuzzySet(points=[[0,1.],[B/2,0]],			term="POOR", high_quality_interpolate=USE_HQ)
		BUDGET_MF = MembershipFunction( [myFS4,myFS5,myFS6], concept="BUDGET")"""

        # h function
        myFS4 = FuzzySet(points=[[B / 2, 0], [B, 1.]], term="HIGH", high_quality_interpolate=USE_HQ)
        myFS5 = FuzzySet(points=[[B / 2, 1.], [0, 0], [B, 0]], term="MEDIUM", high_quality_interpolate=USE_HQ)
        myFS6 = FuzzySet(points=[[0, 1.], [B / 2, 0]], term="POOR", high_quality_interpolate=USE_HQ)
        BUDGET_MF = MembershipFunction([myFS4, myFS5, myFS6], concept="BUDGET")
        # h function

        # C

        """A3 = self.MaxIterations-self.IterationsWithVelocityLessThanAvg
		s = int(math.sqrt(self.dimensions))
		A1 = math.pow(A3,1/s)
		A2 = math.pow(A3,2/s)
		myFS7 = FuzzySet(points=[[0,1], [A1,0]],		term="LOW", high_quality_interpolate=USE_HQ)
		myFS8 = FuzzySet(points=[[A1/2,0],[A2,1.],[A3,0]],	term="MEDIUM", high_quality_interpolate=USE_HQ)
		myFS9 = FuzzySet(points=[[A2,0],[A3,1.]],			term="HIGH", high_quality_interpolate=USE_HQ)
		ITERATIONS_STALLING_MF = MembershipFunction( [myFS7,myFS8,myFS9], concept="ITERATIONS_STALLING")"""

        """A3 = self.MaxIterations #-self.IterationsWithVelocityLessThanAvg
		s = int(math.sqrt(self.dimensions))
		A1 = math.pow(A3,1/s)
		A2 = math.pow(A3,2/s)
		myFS7 = FuzzySet(points=[[0,1], [A1,0]],		term="LOW", high_quality_interpolate=USE_HQ)
		myFS8 = FuzzySet(points=[[0,0],[A1,1.],[(A1+A2)/2,1],[A2,0]],	term="MEDIUM", high_quality_interpolate=USE_HQ)
		myFS9 = FuzzySet(points=[[(A1+A2)/2,0],[A2,1.],[A3,1]],			term="HIGH", high_quality_interpolate=USE_HQ)
		ITERATIONS_STALLING_MF = MembershipFunction( [myFS7,myFS8,myFS9], concept="ITERATIONS_STALLING")
		"""

        # f
        A3 = self.MaxIterations  # -self.IterationsWithVelocityLessThanAvg
        s = int(math.sqrt(self.dimensions))
        A1 = math.pow(A3, 1 / s) / A3
        A2 = math.pow(A3, 2 / s) / A3
        myFS7 = FuzzySet(points=[[0, 1], [A1, 0]], term="LOW", high_quality_interpolate=USE_HQ)
        myFS8 = FuzzySet(points=[[0, 0], [A1, 1.], [(A1 + A2) / 2, 1], [A2, 0]], term="MEDIUM",
                         high_quality_interpolate=USE_HQ)
        myFS9 = FuzzySet(points=[[(A1 + A2) / 2, 0], [A2, 1.], [1, 1]], term="HIGH", high_quality_interpolate=USE_HQ)
        ITERATIONS_STALLING_MF = MembershipFunction([myFS7, myFS8, myFS9], concept="ITERATIONS_STALLING")

        # C

        myR1 = FuzzyRule(IF(DISTANCE_MF, "CLOSE"), THEN("RADIUS", LOW_RADIUS), comment="Rule radius close distance")
        myR2 = FuzzyRule(IF(DISTANCE_MF, "MEDIUM"), THEN("RADIUS", MEDIUM_RADIUS),
                         comment="Rule radius medium distance")
        myR3 = FuzzyRule(IF(DISTANCE_MF, "FAR"), THEN("RADIUS", HIGH_RADIUS), comment="Rule radius far distance")

        """myR4b = FuzzyRule( IF(BUDGET_MF, "HIGH"),  THEN("RADIUS", HIGH_RADIUS), comment="Rule radius high budget" )
		myR5b = FuzzyRule( IF(BUDGET_MF, "MEDIUM"),  THEN("RADIUS", MEDIUM_RADIUS), comment="Rule radius medium budget" )
		myR6b = FuzzyRule( IF(BUDGET_MF, "POOR"),  THEN("RADIUS", LOW_RADIUS), comment="Rule radius poor budget" )"""
        # f function
        myR4 = FuzzyRule(IF(BUDGET_MF, "HIGH"), THEN("ALPHA", HIGH_ALPHA), comment="Rule alpha high budget")
        myR5 = FuzzyRule(IF(BUDGET_MF, "MEDIUM"), THEN("ALPHA", MEDIUM_ALPHA), comment="Rule alpha medium budget")
        myR6 = FuzzyRule(IF(BUDGET_MF, "POOR"), THEN("ALPHA", LOW_ALPHA), comment="Rule alpha poor budget")

        # h function
        """myR4 = FuzzyRule( IF(BUDGET_MF, "HIGH"),  THEN("GAMMA", LOW_GAMMA), comment="Rule gamma high budget" )
		myR5 = FuzzyRule( IF(BUDGET_MF, "MEDIUM"),  THEN("GAMMA", MEDIUM_GAMMA), comment="Rule gamma medium budget" )
		myR6 = FuzzyRule( IF(BUDGET_MF, "POOR"),  THEN("GAMMA", HIGH_GAMMA), comment="Rule gamma poor budget" )"""

        # g function
        """myR4 = FuzzyRule( IF(BUDGET_MF, "HIGH"),  THEN("BETA", HIGH_BETA), comment="Rule beta high budget" )
		myR5 = FuzzyRule( IF(BUDGET_MF, "MEDIUM"),  THEN("BETA", MEDIUM_BETA), comment="Rule beta medium budget" )
		myR6 = FuzzyRule( IF(BUDGET_MF, "POOR"),  THEN("BETA", LOW_BETA), comment="Rule beta poor budget" )"""

        # C

        myR7 = FuzzyRule(IF(ITERATIONS_STALLING_MF, "LOW"), THEN("RADIUS", LOW_RADIUS),
                         comment="Rule radius high n iter stalling")
        myR8 = FuzzyRule(IF(ITERATIONS_STALLING_MF, "MEDIUM"), THEN("RADIUS", MEDIUM_RADIUS),
                         comment="Rule radius medium n iter stalling")
        myR9 = FuzzyRule(IF(ITERATIONS_STALLING_MF, "HIGH"), THEN("RADIUS", HIGH_RADIUS),
                         comment="Rule radius low n iter stalling")
        # C

        FR.add_rules([myR1, myR2, myR3, myR4, myR5, myR6, myR7, myR8, myR9])  # myR4b,myR5b,myR6b   myR7,myR8,myR9 #C
        return FR

    def CreateFuzzyReasoner(self, max_delta):

        FR = FuzzyReasoner()

        p1 = max_delta * self.MDP1
        p2 = max_delta * self.MDP2
        p3 = max_delta * self.MDP3

        USE_HQ = False

        myFS1 = FuzzySet(points=[[0, 0], [1., 1.]], term="WORSE", high_quality_interpolate=USE_HQ)
        myFS2 = FuzzySet(points=[[-1., 0], [0, 1.], [1., 0]], term="SAME", high_quality_interpolate=USE_HQ)
        myFS3 = FuzzySet(points=[[-1., 1.], [0, 0]], term="BETTER", high_quality_interpolate=USE_HQ)
        PHI_MF = MembershipFunction([myFS1, myFS2, myFS3], concept="PHI")

        myFS4 = FuzzySet(points=[[0, 1.], [p1, 1.], [p2, 0]], term="SAME", high_quality_interpolate=USE_HQ)
        myFS5 = FuzzySet(points=[[p1, 0], [p2, 1.], [p3, 0]], term="NEAR", high_quality_interpolate=USE_HQ)
        myFS6 = FuzzySet(points=[[p2, 0], [p3, 1.], [max_delta, 1.]], term="FAR", high_quality_interpolate=USE_HQ)
        DELTA_MF = MembershipFunction([myFS4, myFS5, myFS6], concept="DELTA")

        myR1 = FuzzyRule(IF(PHI_MF, "WORSE"), THEN("INERTIA", self.LOW_INERTIA), comment="Rule inertia worse phi")
        myR2 = FuzzyRule(IF(PHI_MF, "WORSE"), THEN("SOCIAL", self.HIGH_SOCIAL), comment="Rule social worse phi")
        myR3 = FuzzyRule(IF(PHI_MF, "WORSE"), THEN("COGNITIVE", self.MEDIUM_COGNITIVE),
                         comment="Rule cognitive worse phi")
        myR4 = FuzzyRule(IF(PHI_MF, "WORSE"), THEN("MINSP", self.HIGH_MINSP), comment="Rule min speed worse phi")
        myR5 = FuzzyRule(IF(PHI_MF, "WORSE"), THEN("MAXSP", self.HIGH_MAXSP), comment="Rule max speed worse phi")

        myR6 = FuzzyRule(IF(PHI_MF, "SAME"), THEN("INERTIA", self.MEDIUM_INERTIA), comment="Rule inertia same phi")
        myR7 = FuzzyRule(IF(PHI_MF, "SAME"), THEN("SOCIAL", self.MEDIUM_SOCIAL), comment="Rule social same phi")
        myR8 = FuzzyRule(IF(PHI_MF, "SAME"), THEN("COGNITIVE", self.MEDIUM_COGNITIVE),
                         comment="Rule cognitive same phi")
        myR9 = FuzzyRule(IF(PHI_MF, "SAME"), THEN("MINSP", self.LOW_MINSP), comment="Rule min speed same phi")
        myR10 = FuzzyRule(IF(PHI_MF, "SAME"), THEN("MAXSP", self.MEDIUM_MAXSP), comment="Rule max speed same phi")

        myR11 = FuzzyRule(IF(PHI_MF, "BETTER"), THEN("INERTIA", self.HIGH_INERTIA), comment="Rule inertia better phi")
        myR12 = FuzzyRule(IF(PHI_MF, "BETTER"), THEN("SOCIAL", self.LOW_SOCIAL), comment="Rule social better phi")
        myR13 = FuzzyRule(IF(PHI_MF, "BETTER"), THEN("COGNITIVE", self.HIGH_COGNITIVE),
                          comment="Rule cognitive better phi")
        myR14 = FuzzyRule(IF(PHI_MF, "BETTER"), THEN("MINSP", self.LOW_MINSP), comment="Rule min speed better phi")
        myR15 = FuzzyRule(IF(PHI_MF, "BETTER"), THEN("MAXSP", self.MEDIUM_MAXSP), comment="Rule max speed better phi")

        myR16 = FuzzyRule(IF(DELTA_MF, "SAME"), THEN("INERTIA", self.LOW_INERTIA), comment="Rule inertia same delta")
        myR17 = FuzzyRule(IF(DELTA_MF, "SAME"), THEN("SOCIAL", self.MEDIUM_SOCIAL), comment="Rule social same delta")
        myR18 = FuzzyRule(IF(DELTA_MF, "SAME"), THEN("COGNITIVE", self.MEDIUM_COGNITIVE),
                          comment="Rule cognitive same delta")
        myR19 = FuzzyRule(IF(DELTA_MF, "SAME"), THEN("MINSP", self.MEDIUM_MINSP), comment="Rule min speed same delta")
        myR20 = FuzzyRule(IF(DELTA_MF, "SAME"), THEN("MAXSP", self.LOW_MAXSP), comment="Rule max speed same delta")

        myR21 = FuzzyRule(IF(DELTA_MF, "NEAR"), THEN("INERTIA", self.MEDIUM_INERTIA), comment="Rule inertia near delta")
        myR22 = FuzzyRule(IF(DELTA_MF, "NEAR"), THEN("SOCIAL", self.LOW_SOCIAL), comment="Rule social near delta")
        myR23 = FuzzyRule(IF(DELTA_MF, "NEAR"), THEN("COGNITIVE", self.MEDIUM_COGNITIVE),
                          comment="Rule cognitive near delta")
        myR24 = FuzzyRule(IF(DELTA_MF, "NEAR"), THEN("MINSP", self.MEDIUM_MINSP), comment="Rule min speed near delta")
        myR25 = FuzzyRule(IF(DELTA_MF, "NEAR"), THEN("MAXSP", self.MEDIUM_MAXSP), comment="Rule max speed near delta")

        myR26 = FuzzyRule(IF(DELTA_MF, "FAR"), THEN("INERTIA", self.LOW_INERTIA), comment="Rule inertia far delta")
        myR27 = FuzzyRule(IF(DELTA_MF, "FAR"), THEN("SOCIAL", self.MEDIUM_SOCIAL), comment="Rule social far delta")
        myR28 = FuzzyRule(IF(DELTA_MF, "FAR"), THEN("COGNITIVE", self.MEDIUM_COGNITIVE),
                          comment="Rule cognitive far delta")
        myR29 = FuzzyRule(IF(DELTA_MF, "FAR"), THEN("MINSP", self.MEDIUM_MINSP), comment="Rule min speed far delta")
        myR30 = FuzzyRule(IF(DELTA_MF, "FAR"), THEN("MAXSP", self.LOW_MAXSP), comment="Rule max speed far delta")

        """
		supp1 = FuzzyRule( IF(DELTA_MF, "SAME"), 	THEN("GAMMA", self.LOW_GAMMA), comment="Supplementary rule for direction inversion (close)")
		supp2 = FuzzyRule( IF(DELTA_MF, "NEAR"), 	THEN("GAMMA", self.MEDIUM_GAMMA), comment="Supplementary rule for direction inversion (med)")
		supp3 = FuzzyRule( IF(DELTA_MF, "FAR"), 	THEN("GAMMA", self.HIGH_GAMMA), comment="Supplementary rule for direction inversion (distant)")
		"""

        FR.add_rules([myR1, myR2, myR3, myR4, myR5, myR6, myR7, myR8, myR9, myR10,
					  myR11, myR12, myR13, myR14, myR15, myR16, myR17, myR18, myR19, myR20,
					  myR21, myR22, myR23, myR24, myR25, myR26, myR27, myR28, myR29, myR30,
					  # supp1, supp2, supp3
					  ])

        return FR

    def UpdateCalculatedFitness(self, verbose=False):
        """
			Calculate the fitness values for each particle according to user's fitness function,
			and then update the settings of each particle.
		"""

        # parallel evaluation
        if self.ParallelFitness:

            ripop = list(map(lambda x: x.X, self.Solutions))

            if self._discrete_cases is not None:

                ripop = [self._convert_prob_to_particle(data) for data in ripop]
                for particle, instance in zip(self.Solutions, ripop):
                    particle._last_discrete_sample = instance

            import numpy as np
            ripop = np.asfarray(ripop)
            ripop = ripop.transpose()
            if self._FITNESS_ARGS is not None:
                all_fitness = self.FITNESS(ripop, self._FITNESS_ARGS)
            else:
                all_fitness = self.FITNESS(ripop)

        # sequential evaluation
        else:
            all_fitness = []
            for s in self.Solutions:
                all_fitness.append(self.call_fitness(s, self._FITNESS_ARGS))

        fr_cogn = "cognitive" in self.enabled_settings
        fr_soci = "social" in self.enabled_settings
        fr_iner = "inertia" in self.enabled_settings
        fr_maxv = "maxvelocity" in self.enabled_settings
        fr_minv = "minvelocity" in self.enabled_settings

        # for each i-th individual "s"...
        for i, s in enumerate(self.Solutions):

            prev = s.CalculatedFitness
            ret = all_fitness[i]
            if s.MagnitudeMovement != 0:
                s.DerivativeFitness = (ret - prev) / s.MagnitudeMovement

            s.NewDerivativeFitness = self.phi(self.EstimatedWorstFitness, prev, ret, s.MagnitudeMovement,
                                              self.MaxDistance)

            if isinstance(ret, list):
                s.CalculatedFitness = ret[0]
                s.Differential = ret[1]
            else:
                s.CalculatedFitness = ret

            ####### TEST #######

            FR = self.CreateFuzzyReasoner(self.MaxDistance)
            FR.set_variable("PHI", s.NewDerivativeFitness)
            FR.set_variable("DELTA", s.DistanceFromBest)
            res = FR.evaluate_rules()

            if fr_cogn:            s.CognitiveFactor = res["COGNITIVE"]
            if fr_soci:            s.SocialFactor = res["SOCIAL"]
            if fr_iner:            s.Inertia = res["INERTIA"]
            if fr_maxv:            s.MaxSpeedMultiplier = res["MAXSP"]
            if fr_minv:            s.MinSpeedMultiplier = res["MINSP"]
        # if fr_gamm:				s.GammaInverter			= res["GAMMA"]

        self._overall_fitness_evaluations += len(self.Solutions)

        # linear population decrease (experimental)
        if "lin_pop_decrease" in self.enabled_settings:
            indices_sorted_fitness = argsort(all_fitness)[::-1]
            nps = self._get_pop_size(NFEcur=self._overall_fitness_evaluations)
            if verbose: print(" * Next population size: %d." % nps)

            ##### WARNING #####
            self.Solutions = [self.Solutions[i] for i in indices_sorted_fitness[:nps]]
        ##### WARNING #####

    def _get_pop_size(self, NFEcur):
        D = self.dimensions
        heuristic_max = int(math.sqrt(D) * math.log(D))
        heuristic_min = int(10 + 2 * math.sqrt(D))
        PSmin = heuristic_min
        PSmax = heuristic_max
        NFEmax = self._FES
        EX = 1
        v = int((float(PSmin - PSmax) / ((NFEmax - PSmax) ** EX)) * (NFEcur - PSmax) ** EX + PSmax)
        if NFEcur + v > NFEmax:
            return NFEmax - NFEcur
        else:
            return v

    def _combine(self, xi, xj, j):
        from numpy.random import uniform
        d = (array(xj) - array(xi)) / 2
        alpha = -1
        beta = (abs(j - 5) - 1) / 3
        c1 = array(xi) - d * (1 + alpha * beta)
        c2 = array(xi) + d * (1 - alpha * beta)
        cnew = c1 + (c2 - c1) * uniform(size=len(xi))

        for d in range(len(xi)):
            if cnew[d] > self.Boundaries[d][1]:
                cnew[d] = self.Boundaries[d][1]
            elif cnew[d] < self.Boundaries[d][0]:
                cnew[d] = self.Boundaries[d][0]

        return cnew

    def _new_recombination(self, particle):
        if self.ParallelFitness:
            raise Exception("Cannot use recombination with parallel fitness")
            exit(-14)

        tobeat = particle.CalculatedFitness

        print("Starting fitness to be beaten: %f" % tobeat)

        all_fitness = [s.CalculatedBestFitness for s in self.Solutions]
        indices_sorted_fitness = argsort(all_fitness)[::-1]
        selected_solutions = [self.Solutions[i] for i in indices_sorted_fitness[:4]]

        index = -1
        best = None
        putative = None
        for n, xj in enumerate(selected_solutions):
            putative = self._combine(particle.X, xj.X, n)
            putfit = self.FITNESS(putative)
            if putfit < tobeat:
                index = n
                tobeat = putfit
                best = putative
            # print(putfit, "*")
            else:
                pass
            # print(putfit)

        if index == -1:
            return particle.X, particle.CalculatedFitness
        else:
            print("Replacing old solution with a new one with fitness %f" % tobeat)
            return list(best), tobeat

    def _get_best_solutions(self, N):
        all_fitness = [s.CalculatedBestFitness for s in self.Solutions]
        indices_sorted_fitness = argsort(all_fitness)[::-1]
        selected_solutions = [self.Solutions[i] for i in indices_sorted_fitness[:N]]
        return selected_solutions

    def _new_recombination2(self, X, trials=100):
        print(" * Trying to reboot...")
        from numpy import average, identity, cov, logspace
        from numpy.random import multivariate_normal
        from matplotlib.pyplot import scatter, show, xlim, ylim, subplots, legend
        # fig, ax = subplots(1,1, figsize=(5,5))
        best_solutions = self._get_best_solutions(int(self.numberofparticles / 3))
        all_sols = []
        for sol in best_solutions:
            all_sols.append(sol.X)
        all_sols = array(all_sols).T
        # print (all_sols)
        com = [average(x, weights=logspace(0, -2, self.numberofparticles / 3)) for x in all_sols]
        cova = cov(all_sols)
        res = multivariate_normal(com, cova, trials)

        if False:
            scatter(all_sols[0], all_sols[1], label="all selected solutions")
            scatter(com[0], com[1], label="weighted average")
            scatter(res.T[0], res.T[1], alpha=0.5, s=10, label="new samples")
            scatter(all_sols[0][0], all_sols[1][0], label="best individual")
            xlim(-100, 100)
            ylim(-100, 100)
            legend()
            show();
            exit()

        for r in res:
            for d in range(len(r)):
                if r[d] > self.Boundaries[d][1]:
                    r[d] = self.Boundaries[d][1]
                elif r[d] < self.Boundaries[d][0]:
                    r[d] = self.Boundaries[d][0]

        allnewfit = [self.FITNESS(r) for r in res]
        best = argmin(allnewfit)

        self._overall_fitness_evaluations += trials

        if allnewfit[best] < X.CalculatedBestFitness:
            return list(res[best]), allnewfit[best]
        else:
            return X.X, X.CalculatedFitness

    def remap(self, point, n):
        boundaries = self.Boundaries
        return (boundaries[n][1] - boundaries[n][0]) * point + boundaries[n][0]

    def normalize(self, point, n):
        boundaries = self.Boundaries
        return (point - boundaries[n][0]) / (boundaries[n][1] - boundaries[n][0])

    def UpdatePositions(self, verbose=False, use_recombination=False):
        """
			Update particles' positions and update the internal information.
		"""

        if self.UseRestart:
            if self.Iterations + self._threshold_local_update < self.MaxIterations:  # do not reboot at the end of the optimization...

                for p in self.Solutions:

                    if p.MarkedForRestart:
                        if use_recombination:
                            p.X, p.CalculatedBestFitness = self._new_recombination2(p)
                            p.CalculatedFitness = p.CalculatedBestFitness
                        else:
                            p.X = self.NewGenerate(zeros(self.dimensions), creation_method=self._used_generation_method)
                        p.B = copy.deepcopy(p.X)
                        p.V = list(zeros(len(p.X)))
                        p.SinceLastLocalUpdate = 0
                        p.MarkedForRestart = False
                        if verbose: print("REBOOT happened")
                    # print ("REBOOT happened")
        dim = self.dimensions
        for p in self.Solutions:

            if not p.can_move(): continue

            prev_pos = p.X[:]
            original_x_c_norm = linalg.norm(array(p.V), ord=2)
            actual_x_c_norm = 0
            actual_center = list(zeros(dim))
            R = 0
            alpha = 1
            gamma = 1
            beta = 1

            BF = 1
            # InBubble = True

            # if not self.UseDilation: #tmp
            #	p.RadiusOverTime.append(p.DistanceFromBest)

            # contained = None
            # error
            # contained = False
            # error

            """counter_switches = 0
			counter_negatives = 0
			counter_positives = 0
			counter_zeros = 0"""

            if self.UseDilation and p.IsStalling:
                # noerror	
                """if p.PreviouslyStalling:
					contained = True
					for n in range(len(p.X)):
						c1 = p.X[n]
						if c1 >= p.PreviousCenter[n]+p.PreviousR:
							contained = False
							p.PreviousR = 0
							#p.PreviousAlpha = 1
							p.PreviousGamma = 1
							p.PreviousCenter[n] = 0
							break"""
                # noerror		
                # p.IterationsStalling += 1
                L = self.L
                B = self._FES
                FR_radius = 0
                """if contained: #thus p.PreviouslyStalling == True
					#extend R
					FR_radius = self.CreateRadiusFuzzyReasoner(p.PreviousR*math.sqrt(2),B,low_value = p.PreviousR)
					# x_c norm depends on the previous center
					actual_x_c_norm = linalg.norm(array(p.X)+array(p.V)-array(p.PreviousCenter), ord=2)
					actual_center = p.PreviousCenter
				else:
					#place a new bubble centered p.X
					FR_radius = self.CreateRadiusFuzzyReasoner(L,B)
					actual_x_c_norm = original_x_c_norm
					actual_center = p.X"""
                # remedy2
                FR_radius = self.CreateRadiusFuzzyReasoner(L, 1)
                actual_x_c_norm = original_x_c_norm
                actual_center = p.X
                FR_radius.set_variable("DISTANCE", p.DistanceFromBest / self.diag)
                FR_radius.set_variable("BUDGET", (self._FES - self._overall_fitness_evaluations) / B)
                # FR_radius.set_variable("ITERATIONS_STALLING",p.IterationsStalling)
                FR_radius.set_variable("ITERATIONS_STALLING", p.IterationsStalling / self.MaxIterations)  # C
                res = FR_radius.evaluate_rules()
                R = res['RADIUS']
                alpha = res['ALPHA']
                # gamma = res['GAMMA']
                # beta = res['BETA']
                p.PreviousR = R
                p.PreviousAlpha = alpha
                # p.PreviousGamma = gamma
                # p.PreviousBeta=beta

                self.Radii.append(R)
                if actual_x_c_norm >= R:
                    # InBubble = False
                    p.R = 0
                    p.alpha = 1
                # p.gamma = 1
                # p.beta=1
                # noerror
                """if actual_x_c_norm >= R:
					InBubble = False
					p.R = 0
					p.gamma = 1
					p.alpha = 1
				else:
					BF = df.h(gamma,actual_x_c_norm/R)"""
            # noerror
            print(
                f"Fun: {self.funname}, Dilation: {self.UseDilation}, Seedn: {self.SeedNumber}, Particle: {p.ID}, Iteration: {self.Iterations}/{self.MaxIterations}, R: {R}")
            dilation_diff = []
            BubbleExceeded = False
            # norm = linalg.norm(array(p.V))
            # BF = df.f(alpha,norm/R)
            # BF = df.h(gamma,norm/R)
            # BF = df.g(beta,norm/R)
            for n in range(len(p.X)):
                c1 = p.X[n]
                c2 = p.V[n]

                tv = c1 + c2

                # error
                # InBubble = False
                # error

                # noerror
                """if self.UseDilation and p.IsStalling and InBubble:
					p.b0[n] = tv
					tv_old = tv
					diff = tv - actual_center[n]
					sign = int(choice([-1,1], 1, p = self.SignProbabilities))
					#tv = actual_center[n] + (BF * (diff/actual_x_c_norm) * R) 
					tv = c1 + c2*sign #actual_center[n] + (BF * (diff/actual_x_c_norm) * R)*sign
					dilation_diff.append(tv-tv_old)
					p.R = R
					p.gamma = gamma
					#p.alpha = alpha"""
                # noerror

                # error
                if self.UseDilation and p.IsStalling:
                    BF = df.f(alpha, c2 / R)
                    # BF = df.h(gamma,c2/R)
                    # BF = df.g(beta,c2/R)
                    p.b0[n] = tv
                    tv_old = tv

                    # diff = tv - c1[n]
                    tv = BF * (c2 / original_x_c_norm) * R + c1
                    # if tv > c1 + R:
                    # BubbleExceeded = True
                    """if sign(float(c2)) != sign(float(BF)):
						#counter_switches += 1
						counter_negatives += 1
					if sign(float(c2)) == sign(float(BF)) == 1.0:
						counter_positives += 1
					if sign(float(c2)) == sign(float(BF)) == -1.0:
						#counter_negatives += 1
						counter_switches += 1
					if sign(float(c2)) == sign(float(BF)) == 0.0:
						counter_zeros += 1"""

                    dilation_diff.append(tv - tv_old)
                    p.R = R
                    p.alpha = alpha
                # p.beta=beta
                # p.gamma = gamma
                # error

                rnd1 = rnd2 = 0
                if tv > self.Boundaries[n][1]:
                    rnd1 = random.random()
                    tv = self.Boundaries[n][1] - rnd1 * c2
                if tv < self.Boundaries[n][0]:
                    rnd2 = random.random()
                    tv = self.Boundaries[n][0] - rnd2 * c2

                # self.NUpdates += 1
                p.PreviousX[n] = p.X[n]
                p.X[n] = tv
            """if BubbleExceeded:
				p.BubbleExceeded = linalg.norm(array(p.X)-array(p.PreviousX)-array([R]*self.Dimensions))
			else:
				p.BubbleExceeded = 0"""
            p.R = R
            p.alpha = alpha
            # p.beta=beta
            # p.gamma = gamma
            dilation_diff_norm = linalg.norm(dilation_diff, ord=2)
            p.DilationEffect = dilation_diff_norm
            p.PreviouslyStalling = p.IsStalling
            p.MagnitudeMovement = linalg.norm(array(p.X) - array(prev_pos), ord=2)
            p.DistanceFromBest = linalg.norm(array(p.X) - array(self.G.X), ord=2)

            # cose mie
            """p.counter_switches = counter_switches
			p.counter_negatives = counter_negatives
			p.counter_positives = counter_positives
			p.counter_zeros = counter_zeros"""

    # logging.info('Particles positions updated.')

    def UpdateVelocities(self):
        """
			Update the velocity of all particles, according to their own settings.
		"""

        for p in self.Solutions:

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
                p.VelocityOverTime.append(newvelocity)

    # logging.info('Particles velocities updated.')

    def TerminationCriterion(self, verbose=False):
        """ 
			This new method for termination criterion verification 
			supports FES exaustion. 
		"""

        if verbose:
            print(" * Iteration:", self.Iterations),
            print("   since last global update:", self.SinceLastGlobalUpdate)

        if self.StopOnGoodFitness == True:
            if self.G.CalculatedFitness < self.GoodFitness:
                if verbose:
                    print(" * Optimal fitness reached", self.G.CalculatedFitness)
                return True

        if self.SinceLastGlobalUpdate > self.MaxNoUpdateIterations:
            if verbose:
                print(" * Maximum number of iterations without a global best update was reached")
            return True

        # print (" * Iteration %d: used %d/%d f.e." % (self.Iterations, self._overall_fitness_evaluations, self._FES))

        if self._FES is not None:
            if self._overall_fitness_evaluations >= self._FES:
                print(" * Budget of fitness evaluations exhausted after %d iterations." % (self.Iterations + 1))
                return True
        else:
            if self.Iterations > self.MaxIterations:
                if verbose:
                    print(" * Maximum iterations reached.")
                return True
            else:
                return False


def calculate_max_distance(interval):
    accum = 0
    for i in interval:
        accum += (i[1] - i[0]) ** 2
    return math.sqrt(accum)


if __name__ == '__main__':
    print("ERROR: please create a new FuzzyPSO object, specify a fitness function and the search space")
