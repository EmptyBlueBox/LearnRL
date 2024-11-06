N_episode = 1000
K_epoch = 3
T_horizon = 32
batch_size = 16

learning_rate = 3e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.1

use_buffer = 0 # 是否使用之前的数据
max_buffer_dis = 5 # buffer 最长距离

model_folder = './PPO_model'
model_name = 'PPO_model'
model_suffix = '2024_05_16_03_14_51'

visualize_folder = './PPO_result'
# visualize_suffix = 'no_buffer'
visualize_suffix = 'best_model'
