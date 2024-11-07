import numpy as np
from gymnasium import Env

class BaseEnv(Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
    
class TestEnv(BaseEnv):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.step(0)
    env.close()