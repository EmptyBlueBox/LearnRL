import numpy as np
# import torch
from gymnasium import Env
from gymnasium.spaces import Box

from isaacsim import SimulationApp

headless = True
headless = False
simulation_app = SimulationApp({"headless": headless})


from omni.isaac.core import World, SimulationContext
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.dynamic_control import _dynamic_control

class BaseEnv(Env):
    def __init__(self, 
                 headless: bool = True, 
                 device: str = 'cuda', 
                 num_envs: int = 64,
                 spacing: float = 2.0,
                 sub_control_freq: int = 5):
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.spacing = spacing
        self.sub_control_freq = sub_control_freq
        
        # initialize the simulation app
        # self.simulation_app = SimulationApp({"headless": headless})
        
        # Initialize the world
        self.world = World()
        self.world.scene.add_default_ground_plane()
        
        # Get SimulationContext
        self.simulation_context = SimulationContext()

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def close(self):
        pass
    
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
            add_reference_to_stage(usd_path=h1_usd_path, prim_path=f"/World/H1_{i}")
        
        #  Initialize the stage and ensure the world is stepped
        self.simulation_context.initialize_physics()
        
        self.h1_system = ArticulationView(prim_paths_expr=f"/World/H1_*", name="h1_view")
        self.world.scene.add(self.h1_system)
        self.h1_system.initialize()
        
        self.simulation_context.play()
        
        # Calculate the initial translation
        self.initial_translation = np.zeros((self.num_envs, 3))
        self.initial_orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.num_envs, 1))
        n = int(np.ceil(np.sqrt(self.num_envs)))
        for i in range(self.num_envs):
            row = i // n
            col = i % n
            self.initial_translation[i] = np.array([
                (col - (n-1)/2) * self.spacing,
                (row - (n-1)/2) * self.spacing,
                3
            ])
            
        # Initialize the state
        print(f"Initialized {self.num_envs} environments")
        
        # Reset root translation
        self.reset()
            
    def step(self, actions):
        self.world.step(render = not self.headless)
        
        self.h1_system.set_joint_position_targets(actions)
        self.h1_system.apply_action(actions)
        
        # # Apply the actions using dynamic control interface
        # dc = _dynamic_control.acquire_dynamic_control_interface()
        
        # for i in range(self.num_envs):
        #     articulation = dc.get_articulation(f"/World/H1_{i}")
            
        #     if i == 0:
        #         num_joints = dc.get_articulation_joint_count(articulation)
        #         num_dofs = dc.get_articulation_dof_count(articulation)
        #         num_bodies = dc.get_articulation_body_count(articulation)
        #         print(f'num_joints: {num_joints}')
        #         print(f'num_dofs: {num_dofs}')
        #         print(f'num_bodies: {num_bodies}')
            
        #     dc.wake_up_articulation(articulation)
        #     dc.set_articulation_dof_position_targets(articulation, actions[i])
            
        for i in range(self.sub_control_freq):
            self.world.step(render=not self.headless)
            
        return self._get_state()
    
    def _get_state(self):
        joint_positions = self.h1_system.get_joint_positions()
        joint_velocities = self.h1_system.get_joint_velocities()
        state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities
        }
        return state_dict
    
    def reset(self):
        # set the initial pose
        self.h1_system.set_world_poses(positions=self.initial_translation, orientations=self.initial_orientations)
        
    def debug(self):
        print(f'Current world pose: {self.h1_system.get_world_poses()}')
        print(f'Current angular velocity: {self.h1_system.get_angular_velocities()}')
        print(f'Current linear velocity: {self.h1_system.get_linear_velocities()}')
        print(f'Current joint positions: {self.h1_system.get_joint_positions()}')
        print(f'Current joint velocities: {self.h1_system.get_joint_velocities()}')

class G1Env(BaseEnv):
    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        

def test_h1():
    num_envs = 9
    num_dof = 37
    env = H1Env(headless=headless, 
                device='cuda', 
                num_envs=num_envs, 
                spacing=3.0,
                sub_control_freq=3)
    for loop in range(1):
        env.reset()
        for time_step in range(60):
            state = env.step(np.zeros((num_envs, num_dof)))
            env.debug()
        env.reset()
        print(f"Loop {loop} finished")
        
    env.close()
    
    simulation_app.close()

def test_g1():
    env = G1Env()
    env.reset()
    env.step(0)
    env.close()

if __name__ == "__main__":
    test_h1()