import gym
import torch
import datetime
import os
from torch.distributions import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PPO_model import PPO
from PPO_hyperparameters import T_horizon, N_episode, model_name, max_buffer_dis, model_folder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make('CartPole-v1')
    model = PPO().to(device)  # 将模型移动到GPU
    # 初始化TensorBoard，添加年月日时间戳
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    writer = SummaryWriter(f'runs/PPO-time_{current_time}')

    for n_epi in tqdm(range(N_episode)):
        s = env.reset()
        if isinstance(s, tuple):
            s = s[0]  # 如果返回的是元组，则提取第一个元素作为状态
        done = False
        episode_reward = 0
        while not done:
            for t in range(T_horizon):
                s_tensor = torch.from_numpy(s).float().to(device)  # 将状态张量移动到GPU
                prob = model.pi(s_tensor)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated  # 将 terminated 和 truncated 逻辑或

                if isinstance(s_prime, tuple):
                    s_prime = s_prime[0]  # 如果返回的是元组，则提取第一个元素作为状态

                # 存储经验
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))

                s = s_prime

                episode_reward += r
                if done:
                    break

            loss1, loss2 = model.train_net()  # 训练网络

        writer.add_scalar('train/Total Loss', loss1+loss2, n_epi)  # 记录损失
        writer.add_scalar('train/Loss 1', loss1, n_epi)  # 记录损失
        writer.add_scalar('train/Loss 2', loss2, n_epi)  # 记录损失
        writer.add_scalar('train/Reward', episode_reward, n_epi)  # 记录每个episode的奖励
        writer.add_scalar('train/buffer size', len(model.data), n_epi) #记录经验池大小

    # 保存模型
    os.makedirs(model_folder, exist_ok=True)
    torch.save(model.state_dict(), model_name+'-'+current_time+'.pth')
    print(f"Model saved to {model_name+'-'+current_time}")

    env.close()
    writer.close()  # 关闭TensorBoard

if __name__ == '__main__':
    main()