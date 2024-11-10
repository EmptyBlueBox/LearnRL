import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from isaacsim import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction

class BaseEnv(Env):
    def __init__(self, 
                 headless: bool = True, 
                 device: str = 'cuda', 
                 num_envs: int = 64,
                 spacing: float = 2.0,
                 control_freq: int = 10):
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.spacing = spacing
        self.control_freq = control_freq
        
        # initialize the simulation app
        self.simulation_app = SimulationApp({"headless": headless})
        
        # initialize the world
        self.world = World()
        self.world.scene.add_default_ground_plane()

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def close(self):
        self.simulation_app.close()
    
    def _get_observation(self, idx):
        pass

    def _compute_reward(self, observation, action):
        pass

    def _check_done(self, idx):
        pass

class H1Env(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # add the robot to the world
        h1_usd_path = "/home/emptybluebox/LearnRL/HumanoidBenchmark/source/asset/g1.usd"
        for i in range(self.num_envs):
            add_reference_to_stage(h1_usd_path, f"/World/H1_{i}")
        self.h1_system = ArticulationView(prim_path=f"/World/H1_[0-{self.num_envs-1}]", name=f"h1_view")
        self.world.scene.add(self.h1_system)
        
        # initialize the initial translation
        self.initial_translation = torch.zeros((self.num_envs, 3), device=self.device)
        n = int(np.ceil(np.sqrt(self.num_envs)))
        for i in range(self.num_envs):
            row = i // n
            col = i % n
            self.initial_translation[i] = torch.tensor([
                (col - (n-1)/2) * self.spacing,
                (row - (n-1)/2) * self.spacing,
                0
            ], device=self.device)
            
        # initialize the state
        self.reset()
            
    def step(self, actions):
        self.world.step(render = not self.headless)
        
        # Apply the actions, use the controller to do the PD control
        for i in range(self.control_freq):
            self.h1_system.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=actions)
            )
    
    def get_state(self):
        joint_positions = self.h1_system.get_joint_positions()
        joint_velocities = self.h1_system.get_joint_velocities()
        state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities
        }
        return state_dict
    
    def reset(self):
        # set the initial pose
        self.h1_system.set_world_pose(position=self.initial_translation, rotation=np.array([0, 0, 0, 1]))
        self.h1_system.set_joint_positions(np.zeros(19))

class G1Env(BaseEnv):
    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        

def test_h1():
    env = H1Env(headless=False, 
                device='cuda', 
                num_envs=1, 
                spacing=2.0)
    for loop in range(10):
        env.reset()
        for time_step in range(100):
            env.step(torch.zeros(19, device=env.device))
            state = env.get_state()
            print(state)
        print(f"Loop {loop} finished")
        
    env.close()

def test_g1():
    env = G1Env()
    env.reset()
    env.step(0)
    env.close()

if __name__ == "__main__":
    test_h1()