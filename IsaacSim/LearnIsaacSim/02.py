# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# 可视化 H1

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.nucleus import get_assets_root_path

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

# assets_root_path = get_assets_root_path()
# if assets_root_path is None:
#     carb.log_error("Could not find Isaac Sim assets folder")
#     simulation_app.close()
#     sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
asset_path = '/home/emptybluebox/LearnRL/IsaacSim/LearnIsaacSim/asset/h1.usd'
# asset_path = '/home/emptybluebox/LearnRL/IsaacSim/LearnIsaacSim/asset/franka_alt_fingers.usd'
add_reference_to_stage(usd_path=asset_path, prim_path="/World/H1_1")
add_reference_to_stage(usd_path=asset_path, prim_path="/World/H1_2")
articulated_system_1 = my_world.scene.add(Robot(prim_path="/World/H1_1", name="my_h1_1"))
articulated_system_2 = my_world.scene.add(Robot(prim_path="/World/H1_2", name="my_h1_2"))


for i in range(1):
    print("resetting...")
    my_world.reset()
    articulated_system_1.set_world_pose(position=np.array([0.0, 2.0, 0.0]) / get_stage_units())
    articulated_system_2.set_world_pose(position=np.array([0.0, -2.0, 0.0]) / get_stage_units())
    articulated_system_1.set_joint_positions(np.ones(19) * 1.5)
    
    for j in range(500):
        my_world.step(render=True)
        if j == 100:
            articulated_system_2.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=np.ones(19) * 1.5)
            )
        if j == 400:
            print("Franka 1's joint positions are: ", articulated_system_1.get_joint_positions())
            print("Franka 2's joint positions are: ", articulated_system_2.get_joint_positions())
            joint_positions_1 = articulated_system_1.get_joint_positions()
            print(joint_positions_1)
    if args.test is True:
        break

simulation_app.close()
