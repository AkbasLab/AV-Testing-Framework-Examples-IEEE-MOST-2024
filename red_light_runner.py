import scenarioxp as sxp
import sim_bug_tools as sbt
import sumo
import traci

import pandas as pd
import numpy as np

from typing import Callable

class TInterTest:
    def __init__(self):
        self._rng = np.random.RandomState(seed=444)
        traci_client = sumo.TIntersectionClient()

        self.TEST_3D = 3
        self.TEST_3D_EXHAUSTIVE = 4
        self.TEST_5D = 5
        self.TEST_5D_MONTE_CARLO = 6

        self._TEST_ID = self.TEST_5D_MONTE_CARLO

        if self._TEST_ID in [self.TEST_3D, self.TEST_3D_EXHAUSTIVE]:
            self._manager = sxp.ScenarioManager(pd.read_csv("params_3d.csv"))
            self._scenario = sumo.TIntersectionScenario3D
        elif self._TEST_ID in [self.TEST_5D, self.TEST_5D_MONTE_CARLO]:
            self._manager = sxp.ScenarioManager(pd.read_csv("params.csv"))
            self._scenario = sumo.TIntersectionScenario

        self._tsc = lambda s : s["collision"] == 1
    

        self._seq_exp_history = []
        self._fs_exp_history = []
        self._brrt_exp_history = []

        if self._TEST_ID == self.TEST_3D_EXHAUSTIVE:
            self.brute_force()
        elif self._TEST_ID == self.TEST_5D_MONTE_CARLO:
            self.monte_carlo()
        elif self._TEST_ID == self.TEST_5D:
            self.tl_5d()

        traci.close()

        # self.save_tests()
        return

    @property
    def manager(self) -> sxp.ScenarioManager:
        return self._manager
    
    @property
    def tsc(self) -> Callable[[pd.Series], bool]:
        return self._tsc

    @property
    def scenario(self) -> sumo.TIntersectionScenario:
        return self._scenario

    @property
    def seq_exp_history(self) -> list[sxp.SequenceExplorer]:
        return self._seq_exp_history

    @property
    def fs_exp_history(self) -> list[sxp.FindSurfaceExplorer]:
        return self._fs_exp_history

    @property
    def brrt_exp_history(self) -> list[sxp.BoundaryRRTExplorer]:
        return self._brrt_exp_history

    def random_seed(self):
        return self._rng.randint(2**32-1)
    
    def monte_carlo(self):
        for i in range(10):
            print(":: ITERATION %d ::" % i)
            self.monte_carlo_iteration(i)
            print()
        traci.close()
        quit()
        return

    def monte_carlo_iteration(self, iteration : int):
        # Common arguments
        kwargs = {
            "scenario_manager" : self.manager,
            "target_score_classifier" : lambda x: False,
            "scenario" : self.scenario
        }

        # Locate a performance envelope
        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.MONTE_CARLO,
            seed = self.random_seed(),
            fast_foward = self.random_seed() % 10000,
            **kwargs
        )

        for i in range(10000):
            if i % 10 == 0:
                print("TEST %d" % i, end="\r")
            seq_exp.step()

        # traci.close()

        seq_exp.params_history.to_csv("out/mc/params_%d.csv" % iteration)
        seq_exp.score_history.to_csv("out/mc/scores_%d.csv" % iteration)

        # quit()
        return 
    
    def tl_5d(self):
        for i in range(10):
            self.tl_5d_iteration(i)
        traci.close()
        quit()
        return

    def tl_5d_iteration(self, iteration : int):
        no_envelope_id = -1

        for i in range(100):
            print(":: Envelope %d ::" % i)
            self.find_and_explore_one_envelope(n_boundary_samples=100)

            # Count so far
            n = sum([len(seq_exp.params_history.index)\
                for seq_exp in self.seq_exp_history])
            n += sum([len(fs_exp.params_history.index)\
                for fs_exp in self.fs_exp_history])
            n += sum([len(brrt_exp.params_history.index)\
                for brrt_exp in self.brrt_exp_history])
            print("\nCOUNT = %d" % n)
            print()
            if n > 10000:
                break
        
        ls_param_df = []
        ls_score_df = []
        for i in range(len(self.seq_exp_history)):
            seq = self.seq_exp_history[i]
            fs = self.fs_exp_history[i]
            brrt = self.brrt_exp_history[i]
            
            ls_param_df.append(
                seq.params_history.assign(envelope_id = no_envelope_id))
            ls_score_df.append(
                seq.score_history.assign(envelope_id = no_envelope_id))

            ls_param_df.append( fs.params_history.assign(envelope_id = i) )
            ls_score_df.append( fs.score_history.assign(envelope_id = i) )

            ls_param_df.append( brrt.params_history.assign(envelope_id = i) )
            ls_score_df.append( brrt.score_history.assign(envelope_id = i) )
            continue
        
        params_df = pd.concat(ls_param_df)
        params_df.to_csv("out/5d/params_%d.csv" % iteration, index=False)
        
        scores_df = pd.concat(ls_score_df)
        scores_df.to_csv("out/5d/scores_%d.csv" % iteration, index=False)

        self._seq_exp_history.clear()
        self._fs_exp_history.clear()
        self._brrt_exp_history.clear()
        return

    def find_and_explore_one_envelope(self, n_boundary_samples : int):
        # Common arguments
        kwargs = {
            "scenario_manager" : self.manager,
            "target_score_classifier" : self.tsc,
            "scenario" : self.scenario
        }

        # Locate a performance envelope
        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.HALTON,
            seed = self.random_seed(),
            fast_foward = self.random_seed() % 10000,
            **kwargs
        )
        i = 0
        while not seq_exp.step():
            i += 1
            print("Locating Envelope: %d" % i, end="\r")
            continue
        print("                                                    ", end="\r")
        
        self._seq_exp_history.append(seq_exp)
        




        # Find the surface of the envelope.
        fs_exp = sxp.FindSurfaceExplorer(
            root = seq_exp._arr_history[-1],
            seed = self.random_seed(),
            **kwargs
        )
        i = 0
        while not fs_exp.step():
            i += 1
            print("Locating Surface: %d" % i, end="\r")
            continue
        print("                                                    ", end="\r")

        self._fs_exp_history.append(fs_exp)



        # follow the boundary
        root = fs_exp._arr_history[-1]
        brrt_exp = sxp.BoundaryRRTExplorer(
            root = root,
            root_n = sxp.orthonormalize(root, fs_exp.v)[0],
            strategy = "e",
            **kwargs
        )
        

        for i in range(n_boundary_samples):
            print("Following Boundary: %d" % (i+1), end="\r")
            brrt_exp.step()
            continue
        

        self._brrt_exp_history.append(brrt_exp)

        
        return

    def brute_force(self):
        # Common arguments
        kwargs = {
            "scenario_manager" : self.manager,
            "target_score_classifier" : self.tsc,
            "scenario" : self.scenario
        }
        
        exp = sxp.ExhaustiveExplorer(**kwargs)

        i = 0
        target = len(exp.all_combinations)
        while not exp.step():
            i += 1
            print("Test %d of %d (%.2f%%)" % (i, target, i/target*100), 
                end="\r")
            continue

        self.brrt_exp_history.append(exp)
        return


    def save_tests(self):
        if self._TEST_ID == self.TEST_3D:
            suffix = "_3d.csv"
        elif self._TEST_ID == self.TEST_3D_EXHAUSTIVE:
            suffix = "_3d_exh.csv"
        elif self._TEST_ID == self.TEST_5D:
            suffix = "_5d.csv"

        params_df = pd.concat(
            [brrt_exp.params_history.assign(envelope_id = i) \
                for i, brrt_exp in enumerate(self.brrt_exp_history)]
        )

        params_df.to_csv("out/params%s" % suffix, sep=",", index=False)

        scores_df = pd.concat(
            [brrt_exp.score_history.assign(envelope_id = i) \
                for i, brrt_exp in enumerate(self.brrt_exp_history)]
        )
        scores_df["is_target"] = scores_df.apply(
            lambda s: int(self.tsc(s)), axis=1)

        scores_df.to_csv("out/scores%s" % suffix, sep=",", index=False)

        return
        
            

if __name__ == "__main__":
    TInterTest()