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
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction


class BaseEnv(Env):
    def __init__(self, 
                 headless: bool = True, 
                 device: str = 'cuda', 
                 num_envs: int = 64,
                 spacing: float = 2.0,
                 sub_control_freq: int = 5,
                 initial_z_translation: float = None,
                 num_dof: int = None,
                 usd_path: str = None,
                 prim_path_expr: str = None):
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.spacing = spacing
        self.sub_control_freq = sub_control_freq
        self.initial_z_translation = initial_z_translation
        self.num_dof = num_dof
        
        # Initialize the world
        self.world = World()
        self.world.scene.add_default_ground_plane()
        
        # Get SimulationContext
        self.simulation_context = SimulationContext()
        
        # Add the robot to the world
        for i in range(self.num_envs):
            add_reference_to_stage(usd_path=usd_path, prim_path=f"/World/{prim_path_expr}_{i}")
        
        # Initialize the stage and ensure the world is stepped
        self.simulation_context.initialize_physics()
        
        self.system = ArticulationView(prim_paths_expr=f"/World/{prim_path_expr}_*", name=f"{prim_path_expr}_view")
        self.world.scene.add(self.system)
        self.system.initialize()
        
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
                self.initial_z_translation
            ])
        self.initial_linear_velocities = np.zeros((self.num_envs, 3))
        self.initial_angular_velocities = np.zeros((self.num_envs, 3))
        self.initial_joint_positions = np.zeros((self.num_envs, self.num_dof))
        self.initial_joint_velocities = np.zeros((self.num_envs, self.num_dof))
        self.initial_joint_efforts = np.zeros((self.num_envs, self.num_dof))
            
        # Initialize the state
        print(f"Initialized {self.num_envs} environments")
        
        # Reset root translation
        self.reset()

    def step(self, actions):
        self.world.step(render = not self.headless)
        
        action = ArticulationAction(joint_positions=actions)
        self.system.apply_action(action)
        
        for i in range(self.sub_control_freq):
            self.world.step(render=not self.headless)
            
        return self._get_state()
    
    def _get_state(self):
        joint_positions = self.system.get_joint_positions()
        joint_velocities = self.system.get_joint_velocities()
        state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities
        }
        return state_dict
    
    def reset(self):
        # Set the initial pose
        self.system.set_world_poses(positions=self.initial_translation, orientations=self.initial_orientations)
        self.system.set_linear_velocities(self.initial_linear_velocities)
        self.system.set_angular_velocities(self.initial_angular_velocities)
        self.system.set_joint_positions(self.initial_joint_positions)
        self.system.set_joint_velocities(self.initial_joint_velocities)
        self.system.set_joint_efforts(self.initial_joint_efforts)
        
    def debug(self):
        print('-'*80)
        print(f'Current world translation: {self.system.get_world_poses()[0]}')
        print(f'Current world orientation: {self.system.get_world_poses()[1]}')
        print(f'Current linear velocity: {self.system.get_linear_velocities()[0]}')
        print(f'Current angular velocity: {self.system.get_angular_velocities()[0]}')
        print(f'Current joint positions: {self.system.get_joint_positions()[0]}')
        print(f'Current joint velocities: {self.system.get_joint_velocities()[0]}')

class H1Env(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(usd_path="/home/emptybluebox/LearnRL/IsaacSim/MinimumIsaacSImUsage/source/asset/h1.usd", 
                         prim_path_expr="H1", **kwargs)

class G1Env(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(usd_path="/home/emptybluebox/LearnRL/IsaacSim/MinimumIsaacSImUsage/source/asset/g1.usd", 
                         prim_path_expr="G1", **kwargs)

def test_h1():
    num_envs = 16
    num_dof = 19
    env = H1Env(headless=headless, 
                device='cuda', 
                num_envs=num_envs, 
                spacing=3.0,
                sub_control_freq=5,
                initial_z_translation=1.05,
                num_dof=num_dof)

    for loop in range(1):
        env.reset()
        for time_step in range(100):
            # state = env.step(np.random.randn(num_envs, num_dof)*2-1)
            state = env.step(np.zeros((num_envs, num_dof)))
            env.debug()
        print(f"Loop {loop} finished")
        
    env.close()

def test_g1():
    num_envs = 16
    num_dof = 37
    env = G1Env(headless=headless, 
                device='cuda', 
                num_envs=num_envs, 
                spacing=3.0,
                sub_control_freq=5,
                initial_z_translation=0.76,
                num_dof=num_dof)

    for loop in range(1):
        env.reset()
        for time_step in range(100):
            # state = env.step(np.random.randn(num_envs, num_dof)*2-1)
            state = env.step(np.zeros((num_envs, num_dof)))
            env.debug()
        print(f"Loop {loop} finished")
        
    env.close()

if __name__ == "__main__":
    test_h1()
    # test_g1()
    
    simulation_app.close()
