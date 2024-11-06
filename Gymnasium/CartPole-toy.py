import gym

env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(20):
    observation = env.reset()
    done = False
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
