from quadrotor_rl_code.quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
import numpy as np
import time
env = QuadrotorEnvironment()

t0 = time.time()
for _ in range(5):
    env.reset()
    for i in range(1000):
        env.step(np.random.uniform(-1, 1, 4))

print(time.time() - t0)