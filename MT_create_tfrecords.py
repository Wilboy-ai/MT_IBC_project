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

np.set_printoptions(precision=32)

#global_entry_target = 1

class MT_SutureOracle(py_policy.PyPolicy):
    """Creates Policy to generate training data"""
    def __init__(self, env):
        print(env.action_space)
        print(env.action_spec())
        print(env.time_step_spec())
        print(env.test())

        super(MT_SutureOracle, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self._n_steps = 0
        print("reset called from MT_SutureOracle init (MT)")
        #self._env.set_target_entry(global_entry_target)
        #self.read_file = f'Entry{global_entry_target}_demo_recording.csv'
        self.reset()
    def reset(self):
        self._env.reset()
        # Get the action space
        self._action_spec = self._env.action_spec() # Do i need this?

    def _action(self, time_step, policy_state):
        #if self._n_steps <= 0:
        #    self._env.init_arm(self.get_csv_line(self.read_file, 0))
        #print(self._n_steps)

        Oracle_demo_action = np.array(self._env.get_O_action(), dtype=np.float32)

        # Get the action space
        #action_spec = self._env.action_spec()
        # Generate a random action within the action space
        #demo_action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape).astype('float32')

        # Round to 7 digits and convert to float32
        #px, py, pz, ox, oy, oz, ow, gripper = self._env.get_O_action()
        #test = [px, py, pz, ox, oy, oz, ow, gripper]
        #print(f'Oracle_demo_action: {Oracle_demo_action} vs. {test}')

        self._n_steps += 1
        return policy_step.PolicyStep(action=Oracle_demo_action)

def create_episodes(dataset_path, num_episodes, index):
    """Create training data"""
    env = suite_gym.load('SurgicalEnv-v2')
    policy = MT_SutureOracle(env)
    env.set_video_title(f'demo_{index}')

    #TODO: Remove this as this is only a test!
    #env.seed(12345)

    observers = []
    observers.append(example_encoding_dataset.TFRecordObserver(dataset_path, policy.collect_data_spec, py_mode=True,
                                                               compress_image=True))
    driver = py_driver.PyDriver(env, policy, observers=observers, max_episodes=num_episodes)
    time_step = env.reset()
    initial_policy_state = policy.get_initial_state(1)

    driver.run(time_step, initial_policy_state)
    
    env.close()

# def create_video():
#     """Create video of the policy defined by the oracle class."""
#     from utils import Mp4VideoWrapper
#
#     np.random.seed(1)
#     seeds = np.random.randint(size=FLAGS.num_videos, low=0, high=1000)
#
#     for seed in seeds:
#         env = suite_gym.load('Particle-v1')
#         env.seed(seed)
#
#         particle_policy = ParticleOracle(env)
#
#         video_path = os.path.join('data', 'videos', 'ttl=7d', 'vid_%d.mp4' % seed)
#         control_frequency = 30
#
#         video_env = Mp4VideoWrapper(env, control_frequency, frame_interval=1, video_filepath=video_path)
#         driver = py_driver.PyDriver(video_env, particle_policy, observers=[], max_episodes=1)
#
#         time_step = video_env.reset()
#
#         initial_policy_state = particle_policy.get_initial_state(1)
#         driver.run(time_step, initial_policy_state)
#         video_env.close()
#         logging.info('Wrote video for seed %d to %s', seed, video_path)


def main(_):
    num_episodes = 1
    for i in range(0, 1): # + list(range(80, 100)): #range(70, 80):
        dataset_path = f'MT_train_data/suture_throw_demo_{i}.tfrecord'
        dataset_split_path = os.path.splitext(dataset_path)
        create_episodes(dataset_path, num_episodes, i)

    #dataset_path = f'MT_train_data/suture_throw_demo_test.tfrecord'
    #dataset_split_path = os.path.splitext(dataset_path)
    #create_episodes(dataset_path, num_episodes, 'test')

    # context = multiprocessing.get_context()
    # jobs = []
    # for i in range(num_jobs):
    #     dataset_path = dataset_split_path[0] + '_%d' % i + dataset_split_path[1]
    #     job = context.Process(target=create_episodes, kwargs={"dataset_path": dataset_path,
    #                                                           "num_episodes": num_episodes})
    #     job.start
    #     jobs.append(job)
    #
    # for job in jobs:
    #     job.join()

if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))

