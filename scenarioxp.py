import abc
from typing import Callable
from copy import copy

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np
import treelib
import rtree

import scipy.stats.qmc as qmc

import matplotlib.pyplot as plt
import os

import sim_bug_tools as sbt


def project(a : float, b : float, n : float, inc : float = None) -> float:
    """
    Project a normal val @n between @a and @b with an discretization 
    increment @inc.
    """
    assert n >= 0 and n <= 1
    assert b >= a

    # If no increment is provided, return the projection
    if inc is None:
        return n * (b - a) + a

    # Otherwise, round to the nearest increment
    n_inc = (b-a) / inc
    
    x = np.round(n_inc * n)
    return min(a + x*inc, b)
    

def normalize(u: np.ndarray):
    return u / np.linalg.norm(u)


def orthonormalize(
        u: np.ndarray, 
        v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates orthonormal vectors given two vectors @u, @v which form a span.

        -- Parameters --
        u, v : np.ndarray
            Two n-d vectors of the same length
        -- Return --
        (un, vn)
            Orthonormal vectors for the span defined by @u, @v
        """
        # u = u.squeeze()
        # v = v.squeeze()

        assert len(u) == len(v)

        # u = u[np.newaxis]
        # v = v[np.newaxis]

        
        un = normalize(u)
        vn = v - np.dot(un, v.T) * un
        vn = normalize(vn)

        if not (np.dot(un, vn.T) < 1e-4):
            return u, v
            # raise Exception("Vectors %s and %s are already orthogonal." % (un, vn))
        return un, vn


def generateRotationMatrix(
        u: np.ndarray, 
        v: np.ndarray
    ) -> Callable[[float], np.ndarray]:
    """
    Creates a function that can construct a matrix that rotates by a given angle.

    Args:
        u, v : ndarray
            The two vectors that represent the span to rotate across.

    Raises:
        Exception: fails if @u and @v aren't vectors or if they have differing
            number of dimensions.

    Returns:
        Callable[[float], ndarray]: A function that returns a rotation matrix
            that rotates that number of degrees using the provided span.
    """
    u = u.squeeze()
    v = v.squeeze()

    if u.shape != v.shape:
        raise Exception("Dimension mismatch...")
    elif len(u.shape) != 1:
        raise Exception("Arguments u and v must be vectors...")

    u, v = orthonormalize(u, v)

    I = np.identity(len(u.T))

    coef_a = v * u.T - u * v.T
    coef_b = u * u.T + v * v.T

    return lambda theta: I + np.sin(theta) * \
        coef_a + (np.cos(theta) - 1) * coef_b












class SampleOutOfBoundsException(Exception):
    "When a boundary Adherer samples out of bounds, this exception may be thrown"

    def __init__(self, msg="Sample was out of bounds!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: {self.msg}>"

class BoundaryLostException(Exception):
    "When a boundary Adherer fails to find the boundary, this exception is thrown"

    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"

class ExplorationCompleteException(Exception):
    "Thrown when an explorer calls step() when exploration logic is complete."

    def __init__(self):
        self.msg = "No further exploration logic exists."
        super().__init__(self.msg)
        return

    def __str__(self):
        return "<ExplorationCompleteException: %s>" % self.msg


class ScenarioManager():
    def __init__(self, params : pd.DataFrame):
        """
        """
        req_indices = ["feat", "min", "max", "inc"]
        assert all([feat in req_indices for feat in params.columns])

        self._params = params

        # Determine the adjusted increment for each feature if min and max
        # were 0 and 1 respectively.
        b = params["max"]
        a = params["min"]
        i = params["inc"]
        params["inc_norm"] = i/(b-a)
        del b, a, i
        return

    @property
    def params(self) -> pd.DataFrame:
        return self._params

    def project(self, arr : np.ndarray) -> pd.Series:
        """
        Projects a normalized array @arr to selected concrete values from
        parameter ranges
        """
        # all values in arr must be in [0,1]
        if not (all(arr >= 0) and all(arr <= 1)):
            raise SampleOutOfBoundsException()

        df = self.params.copy() \
            .assign(n = arr)
        
        projected = df.apply(
            # lambda s: project(s["min"], s["max"], s["n"]),#, s["inc"]), 
            lambda s: project(s["min"], s["max"], s["n"], s["inc"]), 
            axis=1
        )

        projected.index = self.params["feat"]
        return projected
        



class Scenario(abc.ABC):
    def __init__(self, params : pd.Series):
        """
        The abstract class for the scenario module.
        The scenario takes @params which are generated from a ScenarioManager.
        """
        assert isinstance(params, pd.Series)
        self._params = params
        return

    @property
    def params(self) -> pd.Series:
        """
        Input configuration for this scenario.
        """
        return self._params

    @abc.abstractproperty
    def score(self) -> pd.Series:
        """
        Scenario score.
        """
        raise NotImplementedError







class Explorer(abc.ABC):
    def __init__(self, 
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        assert isinstance(scenario_manager, ScenarioManager)
        assert isinstance(scenario, Callable)
        assert isinstance(target_score_classifier, Callable)

        self._scenario_manager = scenario_manager
        self._scenario = scenario
        self._target_score_classifier = target_score_classifier

        self._arr_history = []
        self._params_history = []
        self._score_history = []
        self._tsc_history = []

        self._stage = 0
        self.STAGE_EXPLORATION_COMPLETE = 12345
        return
    
    @property
    def arr_history(self) -> np.ndarray:
        return np.array(self._arr_history)

    @property
    def params_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._params_history)

    @property
    def score_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._score_history)

    @property
    def tsc_history(self) -> np.ndarray:
        return np.array(self._tsc_history)

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def scenario_manager(self) -> ScenarioManager:
        return self._scenario_manager

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def target_score_classifier(self) -> Callable[[pd.Series], bool]:
        return self._target_score_classifier

    @abc.abstractmethod
    def next_arr(self) -> np.ndarray:
        """
        This gets the next va
        """
        raise NotImplementedError

    def step(self) -> bool:
        """
        Perform one exploration step.
        Returns if the scenario test is in the target score set.
        """
        if self.stage == self.STAGE_EXPLORATION_COMPLETE:
            raise ExplorationCompleteException

        arr = self.next_arr()                        # Long walk
        params = self._scenario_manager.project(arr) # Generate paramas
        test = self._scenario(params)                # Run scenario
        is_target_score = self._target_score_classifier(test.score)

        self._arr_history.append(arr)
        self._params_history.append(params)
        self._score_history.append(test.score)
        self._tsc_history.append(is_target_score)
        return is_target_score

    def concat_history(self, explorer):
        """
        Combines history of @explorer to self.
        """
        [self._arr_history.append(arr) for arr in explorer._arr_history]
        [self._params_history.append(params) \
            for params in explorer._params_history]
        [self._score_history.append(score) \
            for score in explorer._score_history]
        [self._tsc_history.append(tsc) for tsc in explorer._tsc_history]
        return


class HistoryExplorer(Explorer):
    def __init__(self, 
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        """
        A blank explorer object for storing history data. 
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)
        return

    def next_arr(self):
        return 

    def step(self):
        return


import itertools

class ExhaustiveExplorer(Explorer):
    
    def __init__(self,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        """
        The brute force explorer, which exhaustively tests all combinations of
        parameters.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)

        # Get discrete nromal inverals
        discrete_intervals = scenario_manager.params["inc_norm"].apply(
            lambda inc : list(np.arange(0,1,inc))
        ).to_list()

        # Add 1 to the to end since np.arange does not.
        [di.append(1) for di in discrete_intervals]

        # Create parameter combinations
        self._all_combinations = [np.array(params) \
            for params in itertools.product(*discrete_intervals)]
        
        self._ptr = 0
        return

    @property
    def all_combinations(self) -> list[np.ndarray]:
        return self._all_combinations

    @property
    def ptr(self) -> int:
        return self._ptr

    def next_arr(self) -> np.ndarray:
        return self._arr

    def step(self) -> bool:
        self._arr = self.all_combinations[self.ptr]
        super().step()
        self._ptr += 1
        if self._ptr >= len(self._all_combinations):
            self._stage = self.STAGE_EXPLORATION_COMPLETE
            return True
        return False


class SequenceExplorer(Explorer):
    MONTE_CARLO = "random"
    HALTON = "halton"
    SOBOL = "sobol"

    def __init__(self, 
        strategy : str,
        seed : int,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool],
        scramble : bool = False,
        fast_foward : int = 0
    ):
        """
        The simplest explorer is the SequenceExplore, which samples the next
        parameter for a test using a quasi-random sequence.

        -- Params --
        strategy : str
            The sampling strategy. "random", "halton", and "sobol" strategies
            are supported. Random uses the numpy generator, while halton and
            sobol use scipy.stats.qmc generators.
        seed : int
            Seed for the rng which scrambls the sequence if @scramble flag is
            used.
        scramble : bool (default=True)
            Scramble the sequence
        fast_forward : int (default=0)
            The sequence will @fast_foward n iterations during initialization.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)
        assert strategy in ["random", "halton", "sobol"]

        d = len(scenario_manager.params.index)

        if strategy == self.MONTE_CARLO:
            seq = np.random.RandomState(seed = seed)
            if fast_foward:
                seq.random(size=d)
        elif strategy == self.HALTON:
            seq = qmc.Halton(d=d, scramble=scramble)
            if fast_foward:
                seq.fast_forward(fast_foward)
        elif strategy == self.SOBOL:
            seq = qmc.Sobol(d=d, scramble=scramble)
            if fast_foward:
                seq.fast_forward(fast_foward)
        else:
            raise NotImplementedError

        self._d = d
        self._seq = seq
        self._strategy = strategy
        return

    def next_arr(self) -> np.ndarray:
        if self._strategy == self.MONTE_CARLO:
            return self._seq.random(size = self._d)
        elif self._strategy in [self.SOBOL, self.HALTON]:
            return self._seq.random(1)[0]
        raise NotImplementedError

    def step(self) -> bool:
        tsc = super().step()
        if tsc:
            self._stage = self.STAGE_EXPLORATION_COMPLETE
        return tsc


class FindSurfaceExplorer(Explorer):
    def __init__(self, 
        root : np.ndarray,
        seed : int,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        """
        Navigates from @root someplace in an target envelope to the surface.

        -- Additional parameters--
        @seed : int
            Seed for the RNG.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)

        self.root = root

        # Jump distance
        self._d = scenario_manager.params["inc_norm"].to_numpy()
        
        # Find the surface
        rng = np.random.RandomState(seed=seed)
        self._v = rng.rand(len(root))
        self._s = self._v * self._d / np.linalg.norm(self._v)
        self._interm = [root]

        self._prev = None
        self._cur = root

        self._stage = 0
        return

    @property
    def v(self) -> np.ndarray:
        return self._v

    def step(self):
        # Reach within d distance from the surface
        if self._stage == 0:
            self._prev = self._cur
            self._interm += [self._prev]
            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            # Stage end condition
            if all(self._cur == self._prev):
                # print("0: At parameter boundary.")
                self._stage = 1
            elif not super().step():
                # print("0: Past parameter boundary.")
                self._stage = 1
            
            return False
            
        # Transition to d/2
        elif self._stage == 1:
            # print("1: Within d distance from surface." )
            self._s *= 0.5
            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            if all(self._cur == self._prev):
                # print("2: At parameter boundary. Done.")
                self._stage = self.STAGE_EXPLORATION_COMPLETE
                return True
            elif not super().step():
                # print("2: Past parameter boundary. Done")
                self._stage = self.STAGE_EXPLORATION_COMPLETE
                return True

            self._stage = 2
            return False
            
        # Get closer until within d/2 distance from surface
        elif self._stage == 2:
            self._prev = self._cur
            self._interm += [self._prev]

            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            if all(self._cur == self._prev):
                # print("2: At parameter boundary. Done.")
                self._stage = self.STAGE_EXPLORATION_COMPLETE
                return True
            elif not super().step():
                # print("2: Past parameter boundary. Done")
                self._stage = self.STAGE_EXPLORATION_COMPLETE
                return True
            return False
            
        raise NotImplemented

    def next_arr(self) -> np.ndarray:
        return self._cur

    def _round_to_limits(
        self,
        arr : np.ndarray, 
        min : np.ndarray, 
        max : np.ndarray
    ) -> np.ndarray:
        """
        Rounds each dimensions in @arr to limits within @min limits and @max limits.
        """
        is_lower = arr < min
        is_higher = arr > max
        for i in range(len(arr)):
            if is_lower[i]:
                arr[i] = min[i]
            elif is_higher[i]:
                arr[i] = max[i]
        return arr





class BoundaryRRTExplorer(Explorer):
    def __init__(self,
    root : np.ndarray,
    root_n : np.ndarray,
    scenario_manager : ScenarioManager,
    scenario : Callable[[pd.Series], Scenario],
    target_score_classifier : Callable[[pd.Series], bool],
    strategy : str = "constant",
    delta_theta : float = 15 * np.pi / 180,
    theta0 : float = 90 * np.pi / 180,
    N : int = 4,
    ): 
        super().__init__(scenario_manager, scenario, target_score_classifier)
        
        # Adherence Factory
        classifier = self._brrt_classifier
        domain = sbt.Domain.normalized(root.shape[0])
        scaler = scenario_manager.params["inc_norm"].to_numpy() * 2

        strategy = strategy.lower()
        assert strategy in \
            ["constant", "exponential", "const", "exp", "c", "e"]

        # Constant
        if strategy in ["constant", "const", "c"]:
            adh_factory = sbt.ConstantAdherenceFactory(
                classifier, domain, scaler, delta_theta, True
            )

        # Exponential
        else:
            adh_factory = sbt.ExponentialAdherenceFactory(
                # classifier, domain, scaler, theta0, r, N, True
                classifier, scaler, theta0, N, domain, True
            )

        # RRT
        self._brrt = sbt.BoundaryRRT(
            sbt.Point(root), root_n, adh_factory
        )

        # Stats
        self._n_boundary_lost_exceptions = 0
        self._n_sample_out_of_bounds_exceptions = 0
        return

    @property
    def brrt(self) -> sbt.BoundaryRRT:
        return self._brrt
    
    @property
    def n_boundary_lost_exceptions(self) -> int:
        return self._n_boundary_lost_exceptions

    @property
    def n_sample_out_of_bounds_exceptions(self) -> int:
        return self._n_sample_out_of_bounds_exceptions

    def _brrt_classifier(self, p : sbt.Point) -> bool:
        self._arr = p.array
        return super().step()

    def next_arr(self) -> np.ndarray:
        return self._arr

    def step(self):
        try: 
            self.brrt.step()
        except sbt.BoundaryLostException:
            self._n_boundary_lost_exceptions += 1
        except sbt.SampleOutOfBoundsException:
            self._n_sample_out_of_bounds_exceptions += 1
        except SampleOutOfBoundsException: 
            self._n_sample_out_of_bounds_exceptions += 1
        return False