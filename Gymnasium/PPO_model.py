import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from PPO_hyperparameters import learning_rate, gamma, lmbda, eps_clip, K_epoch, batch_size, max_buffer_dis, use_buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.current_iteration = 0

        # 定义神经网络层
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)  # 增加一个隐藏层
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

        # 将模型移动到GPU上
        self.to(device)

        # 定义优化器和学习率调度器
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def pi(self, x, softmax_dim=0):
        # 策略网络，输出动作的概率分布
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 增加一个激活函数
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        # 值网络，输出状态值
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 增加一个激活函数
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        # 存储经验并记录迭代次数
        self.data.append((transition, self.current_iteration))

    def make_batch(self):
        # 将存储的经验转换为训练所需的批次
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        # 筛选有效数据并删除无效数据
        new_data = []
        for transition, iteration in self.data:
            if self.current_iteration - iteration <= max_buffer_dis:  # 保留迭代次数距离不超过 max_buffer_dis 的数据
                s, a, r, s_prime, prob_a, done = transition
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                done_lst.append([0 if done else 1])
                new_data.append((transition, iteration))  # 保留有效数据

        # 更新数据存储，删除无效数据
        if use_buffer:
            self.data = new_data
        else:
            self.data = []

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), \
            torch.tensor(a_lst).to(device), \
            torch.tensor(r_lst).to(device), \
            torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
            torch.tensor(done_lst, dtype=torch.float).to(device), \
            torch.tensor(prob_a_lst).to(device)
            
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        dataset = torch.utils.data.TensorDataset(s, a, r, s_prime, done_mask, prob_a)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(K_epoch):
            for batch in data_loader:
                s, a, r, s_prime, done_mask, prob_a = batch

                # 计算TD目标
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)

                # 计算优势函数
                advantages = delta.detach()

                # 计算策略和值函数的损失
                pi = self.pi(s, softmax_dim=1)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
                loss1 = -torch.min(surr1, surr2)
                loss2 = F.smooth_l1_loss(self.v(s), td_target.detach())
                loss = loss1 + loss2

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.scheduler.step()  # 更新学习率
        self.current_iteration += 1  # 更新迭代次数
        return loss1.mean().item(), loss2.mean().item()  # 分别返回损失值
