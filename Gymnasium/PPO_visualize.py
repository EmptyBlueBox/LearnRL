import gym
import torch
from torch.distributions import Categorical
import imageio
import os
from PPO_model import PPO
from PPO_hyperparameters import model_folder, model_name, model_suffix, visualize_folder, visualize_suffix

def visualize_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    model = PPO().to(device)  # 将模型移动到GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]  # 如果返回的是元组，则提取第一个元素作为状态
    done = False
    episode_reward = 0
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        s_tensor = torch.from_numpy(s).float().to(device)  # 将状态张量移动到GPU
        prob = model.pi(s_tensor)
        m = Categorical(prob)
        a = m.sample().item()
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        episode_reward += r

    env.close()
    print(f"Episode reward: {episode_reward}")

    # 保存视频
    if os.path.exists(visualize_folder) is False:
        os.makedirs(visualize_folder)
    save_frames_as_gif(frames, f'{visualize_folder}/PPO_cartpole-{visualize_suffix}.gif')

def save_frames_as_gif(frames, path):
    imageio.mimsave(path, frames, fps=30)
    print(f"Video saved to {path}")

if __name__ == '__main__':
    visualize_model(f'{model_folder}/{model_name}-{model_suffix}.pth')