import random
import functools
import os
import numpy as np
from absl import app, flags, logging
from tf_agents.policies import py_policy
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import policy_step
from tf_agents.utils import example_encoding_dataset
import MT_Simple_AccelNet_env
#from tf_agents.specs.tensor_spec import from_space
import csv
import random
import time

np.set_printoptions(precision=32)

#global_entry_target = 1

class MT_SutureOracle(py_policy.PyPolicy):
    """Creates Policy to generate training data"""
    def __init__(self, env):
        super(MT_SutureOracle, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self._n_steps = 0
        print("reset called from MT_SutureOracle init (MT)")
        self.reset()
    def reset(self):
        self._env.reset()
        # Get the action space
        self._action_spec = self._env.action_spec() # Do i need this?

    def _action(self, time_step, policy_state):
        Oracle_demo_action = np.array(self._env.get_O_action(), dtype=np.float32)

        self._n_steps += 1
        return policy_step.PolicyStep(action=Oracle_demo_action)

def create_episodes(dataset_path, num_episodes, index):
    """Create training data"""

    # Retry Loading environment logic
    # It doesn't always connect to the launch_crtk_interface, so you have to retry, idk how to fix it..
    for i in range(0, 5):
        try:
            env = suite_gym.load('SurgicalEnv-v2')
            break
        except Exception:
            print(f'Failed to load SurgicalEnv-v2, will try again ({i}/5)')
            time.sleep(1)
            continue

    policy = MT_SutureOracle(env)
    env.set_video_title(f'Suture_demo_{index}')
    env.set_mode_demo()

    observers = []
    observers.append(example_encoding_dataset.TFRecordObserver(dataset_path, policy.collect_data_spec, py_mode=True,
                                                               compress_image=True))
    driver = py_driver.PyDriver(env, policy, observers=observers, max_episodes=num_episodes)
    time_step = env.reset()
    initial_policy_state = policy.get_initial_state(1)

    driver.run(time_step, initial_policy_state)
    
    env.close()

def main(_):
    num_episodes = 1
    for i in [10]: #range(32, 40):
        dataset_path = f'LEVEL_0/suture_throw_demo_{i}.tfrecord'
        create_episodes(dataset_path, num_episodes, i)

if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))
