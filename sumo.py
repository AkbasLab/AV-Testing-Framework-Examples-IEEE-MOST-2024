import shutil
import warnings
if shutil.which("sumo") is None:
    warnings.warn("Cannot find sumo/tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/")
import os

import traci
import traci.constants as tc
import traci.exceptions

import scenarioxp as sxp
import shapes
# import physics
import pandas as pd
import re
import numpy as np


___SUMO_FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
___SUMO_MAP_DIR___ = "%s/map" % ___SUMO_FILE_DIR__
___INIT_SUMO_STATE_FN___ = "%s/init-state.xml" % ___SUMO_MAP_DIR___


def mps2kph(mps : float) -> float:
    return 3.6 * mps

def kph2mps(kph : float) -> float:
    return kph/3.6

class RGBA:
    light_blue = (12,158,236,255)
    rosey_red = (244,52,84,255)



class TraCIClient:
    def __init__(self, config : dict, priority : int = 1):
        """
        Barebones TraCI client.

        --- Parameters ---
        priority : int
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.
        """
        
        self._config = config
        self._priority = priority
        

        self.connect()
        return

    @property
    def priority(self) -> int:
        """
        Priority of TraCI client.
        """
        return self._priority

    @property
    def config(self) -> dict:
        """
        SUMO arguments stored as a python dictionary.
        """
        return self._config

    def run_to_end(self):
        """
        Runs the client until the end.
        """
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # more traci commands
        return

    def close(self):
        """
        Closes the client.
        """
        traci.close()
        return


    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            cmd = []

            for key, val in self.config.items():
                if key == "gui":
                    sumo = "sumo"
                    if val: sumo +="-gui"
                    cmd.append(sumo)
                    continue
                
                if key == "--remote-port":
                    continue

                cmd.append(key)
                if val != "":
                    cmd.append(str(val))
                continue

            traci.start(cmd,port=self.config["--remote-port"])
            traci.setOrder(self.priority)
            return
        
        # Initialize every client after the first.
        traci.init(port=self.config["--remote-port"])
        traci.setOrder(self.priority)
        return    

  

class TIntersectionClient(TraCIClient):
    def __init__(self):
        
        map_dir = ___SUMO_MAP_DIR___
        self._init_state_fn = ___INIT_SUMO_STATE_FN___
        self._error_log_fn = "%s/error.txt" % map_dir
        config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/T-inter.net.xml" % map_dir,

            # Logging
            "--error-log" : self._error_log_fn,
            # "--log" : "%s/log.txt" % map_dir,

            # Quiet Mode
            "--no-warnings" : "",
            "--no-step-log" : "",

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 50,
            "--start" : "",
            "--quit-on-end" : "",
            "--gui-settings-file" : "%s/gui.xml" % map_dir,

            # RNG
            "--seed" : 333,
            
            # Step length
            "--default.action-step-length" : 0.1,
            "--step-length" : 0.1,

        }

        super().__init__(config)
        traci.simulation.saveState(self._init_state_fn)
        return

    @property
    def init_state_fn(self) -> str:
        return self._init_state_fn

    
    
class TIntersectionScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        super().__init__(params)

        # Revert simulation state 
        traci.simulation.loadState(___INIT_SUMO_STATE_FN___)

        # Constants
        self.DUT = "dut"
        self.FOE = "foe"
        
        # Place actors
        self._add_vehicles()
        

        # Init Score
        self._score = pd.Series({
            "collision" : 0,
            "dtc" : -1 
        })

        # Run simulation
        while traci.simulation.getMinExpectedNumber() > 0:
            try:
                self._update_score()
            except traci.exceptions.TraCIException:
                pass
            if traci.vehicle.getIDCount() != 2:
                break
            traci.simulationStep()

        return


    @property
    def score(self) -> pd.Series:
        return self._score


    def _add_vehicles(self):
        """
        The DUT will make a left turn at the green light.
        """
        traci.route.add("dut_left", ["NB","-EB"])
        traci.vehicle.add(self.DUT, "dut_left")
        traci.vehicle.setColor(self.DUT, RGBA.light_blue)

        # Set Decel and Emergency Decel
        decel = self.params["dut_decel"]
        traci.vehicle.setDecel(self.DUT, decel)
        traci.vehicle.setEmergencyDecel(self.DUT, 2 * decel)

        # Warmup DUT
        dut_accel = traci.vehicle.getAccel(self.DUT)
        speed = kph2mps(self.params["dut_speed"])
        traci.vehicle.setAccel(self.DUT, 1000)
        traci.vehicle.setMaxSpeed(self.DUT, speed)
        traci.vehicle.setSpeed(self.DUT, speed)



        """
        The FOE will run the redlight and travel straight in a colliding path
        with the DUT. 
        """
        traci.route.add("foe_straight", ["WB","-EB"])
        traci.vehicle.add(self.FOE, "foe_straight")
        traci.vehicle.setSpeedMode(self.FOE, int(0b00001))
        traci.vehicle.setParameter(
            self.FOE, "junctionModel.ignoreIDs", self.DUT)
        traci.vehicle.setColor(self.FOE, RGBA.rosey_red)

        # Warmup FOE
        foe_accel = traci.vehicle.getAccel(self.FOE)
        speed = kph2mps(self.params["foe_speed"])
        traci.vehicle.setAccel(self.FOE, 1000)
        traci.vehicle.setMaxSpeed(self.FOE, speed)
        traci.vehicle.setSpeed(self.FOE, speed)
        

    
        

        # Spawn vehicles
        traci.simulationStep()

        # Get up to speed
        traci.simulationStep()

        # Restore Accel and Move to stop line.
        traci.vehicle.setAccel(self.DUT, dut_accel)
        pos = 105.6 - self.params["dut_dist"]
        traci.vehicle.moveTo(self.DUT, "NB_0", pos)

        traci.vehicle.setAccel(self.FOE, foe_accel)
        pos = 105.6 - self.params["foe_dist"]
        traci.vehicle.moveTo(self.FOE, "WB_0", pos)

        return


    def _update_score(self):
        # Check for colliding vehicles
        if traci.simulation.getCollidingVehiclesNumber():
            self._score["collision"] = 1
            self._score["dtc"] = 0
            return
            
        # Both DUT and FOE should exist
        if not traci.vehicle.getIDCount() == 2:
            return
        
        # DUT and FOE should both be in the junction
        dut_lid = traci.vehicle.getLaneID(self.DUT)
        foe_lid = traci.vehicle.getLaneID(self.FOE)

        if "J0" in dut_lid and "J0" in foe_lid:
            lane_len = {
                ":J0_0_0" : 14.40,
                ":J0_3_0" : 14.19
            }
            dut_lane_pos = traci.vehicle.getLanePosition(self.DUT)
            foe_lane_pos = traci.vehicle.getLanePosition(self.FOE)
            dtc = lane_len[dut_lid] - dut_lane_pos \
                + lane_len[foe_lid] - foe_lane_pos

            if self._score["dtc"] == -1:
                self._score["dtc"] = np.round(dtc, 3)
            else:
                self._score["dtc"] = np.round(
                        min(self._score["dtc"], dtc), 
                        3
                    )

            
        return
   
    


    

class TIntersectionScenario3D(sxp.Scenario):
    def __init__(self, params : pd.Series):
        super().__init__(params)

        # Revert simulation state 
        traci.simulation.loadState(___INIT_SUMO_STATE_FN___)

        # Constants
        self.DUT = "dut"
        self.FOE = "foe"
        
        # Place actors
        self._add_vehicles()
        

        # Init Score
        self._score = pd.Series({
            "collision" : 0,
            "dtc" : -1 
        })

        # Run simulation
        while traci.simulation.getMinExpectedNumber() > 0:
            try:
                self._update_score()
            except traci.exceptions.TraCIException:
                pass
            if traci.vehicle.getIDCount() != 2:
                break
            traci.simulationStep()

        return


    @property
    def score(self) -> pd.Series:
        return self._score


    def _add_vehicles(self):
        """
        The DUT will make a left turn at the green light.
        """
        traci.route.add("dut_left", ["NB","-EB"])
        traci.vehicle.add(self.DUT, "dut_left")
        traci.vehicle.setColor(self.DUT, RGBA.light_blue)

        # Set Decel and Emergency Decel
        decel = self.params["dut_decel"]
        traci.vehicle.setDecel(self.DUT, decel)
        traci.vehicle.setEmergencyDecel(self.DUT, 2 * decel)

        # Warmup DUT
        dut_accel = traci.vehicle.getAccel(self.DUT)
        speed = kph2mps(self.params["dut_speed"])
        traci.vehicle.setAccel(self.DUT, 1000)
        traci.vehicle.setMaxSpeed(self.DUT, speed)
        traci.vehicle.setSpeed(self.DUT, speed)



        """
        The FOE will run the redlight and travel straight in a colliding path
        with the DUT. 
        """
        traci.route.add("foe_straight", ["WB","-EB"])
        traci.vehicle.add(self.FOE, "foe_straight")
        traci.vehicle.setSpeedMode(self.FOE, int(0b00001))
        traci.vehicle.setParameter(
            self.FOE, "junctionModel.ignoreIDs", self.DUT)
        traci.vehicle.setColor(self.FOE, RGBA.rosey_red)

        # Warmup FOE
        foe_accel = traci.vehicle.getAccel(self.FOE)
        speed = kph2mps(self.params["foe_speed"])
        traci.vehicle.setAccel(self.FOE, 1000)
        traci.vehicle.setMaxSpeed(self.FOE, speed)
        traci.vehicle.setSpeed(self.FOE, speed)
        

    
        

        # Spawn vehicles
        traci.simulationStep()

        # Get up to speed
        traci.simulationStep()

        # Restore Accel and Move to stop line.
        traci.vehicle.setAccel(self.DUT, dut_accel)
        pos = 105.6# - self.params["dut_dist"]
        traci.vehicle.moveTo(self.DUT, "NB_0", pos)

        traci.vehicle.setAccel(self.FOE, foe_accel)
        pos = 105.6# - self.params["foe_dist"]
        traci.vehicle.moveTo(self.FOE, "WB_0", pos)

        return


    def _update_score(self):
        # Check for colliding vehicles
        if traci.simulation.getCollidingVehiclesNumber():
            self._score["collision"] = 1
            self._score["dtc"] = 0
            return
            
        # Both DUT and FOE should exist
        if not traci.vehicle.getIDCount() == 2:
            return
        
        # DUT and FOE should both be in the junction
        dut_lid = traci.vehicle.getLaneID(self.DUT)
        foe_lid = traci.vehicle.getLaneID(self.FOE)

        if "J0" in dut_lid and "J0" in foe_lid:
            lane_len = {
                ":J0_0_0" : 14.40,
                ":J0_3_0" : 14.19
            }
            dut_lane_pos = traci.vehicle.getLanePosition(self.DUT)
            foe_lane_pos = traci.vehicle.getLanePosition(self.FOE)
            dtc = lane_len[dut_lid] - dut_lane_pos \
                + lane_len[foe_lid] - foe_lane_pos

            if self._score["dtc"] == -1:
                self._score["dtc"] = np.round(dtc, 3)
            else:
                self._score["dtc"] = np.round(
                        min(self._score["dtc"], dtc), 
                        3
                    )

            
        return
   
    

