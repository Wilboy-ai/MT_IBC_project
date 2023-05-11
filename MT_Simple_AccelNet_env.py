import collections

import gym
from gym import spaces
from gym.envs import registration
import numpy as np
import tf_agents.environments.py_environment as py_environment
import numpy as np
import cv2
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import BoundedArraySpec
from MT_RosAmbfCommChannel import RosAmbfCommChannel
import time
import datetime
import random
from tf_agents.metrics import py_metrics
from tf_agents.utils import nest_utils
import csv

np.set_printoptions(precision=32)

class SurgicalEnv(gym.Env):
    def get_metrics(self, num_episodes): # TODO: Write the get_metrics function
        metrics = [AverageNeedleEntryDistance(self, buffer_size=num_episodes), AverageGraspSteps(self, buffer_size=num_episodes)]
        success_metric = metrics[-1]
        return metrics, success_metric
    def __init__(self, n_steps=25, repeat_actions=30):
        self._comm_channel = RosAmbfCommChannel()
        self._video = True
        self._frames = []
        self._state = None
        self.steps = 0
        self.n_steps = n_steps
        self.repeat_actions = repeat_actions
        self._current_time_step = None
        self.video_title = "Unknown"
        self.needle_entry_dist = 999
        self.grasp_steps = 0
        self.optimal_action_error = 0
        self.done = False
        self._total_reward = 0
        self.mode = "TRAIN"

        self.csv_file = f'csv_files/unknown.csv'


        # Define the action space
        self.n_dim = 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_dim+1,), dtype=np.float32)
        self.observation_space = self._create_observation_space()
        print("reset called from SurgicalEnv init (MT)")
        self.reset()
    def _create_observation_space(self):
        obs_dict = collections.OrderedDict(
            #psm1=spaces.Box(low=-1.0, high=1.0, shape=(self.n_dim,), dtype=np.float32),
            psm2=spaces.Box(low=-1.0, high=1.0, shape=(self.n_dim,), dtype=np.float32),
            needle=spaces.Box(low=-1.0, high=1.0, shape=(self.n_dim,), dtype=np.float32),
            #entry=spaces.Box(low=-1.0, high=1.0, shape=(self.n_dim,), dtype=np.float32),
            entry=spaces.Box(low=1, high=4, shape=(1,), dtype=np.float32),
            #image=spaces.Box(low=0.0, high=1.0, shape=(64800,), dtype=np.float32) #777600 # stereo image
            #image=spaces.Box(low=0.0, high=255.0, shape=(68, 120, 3), dtype=np.float32) # Mono image pixelEBM
        )
        return spaces.Dict(obs_dict)

    def seed(self, seed=None):
        print(f'random seed set in env: {seed}')
        # Remember to call this function in the beginning of the training session to get consistent evals
        random.seed(seed)
        self._comm_channel.set_seed(seed)

    def set_video_title(self, title):
        self.video_title = title

    def set_csv_file(self, title):
        self.csv_file = title


    def test(self):
        print("this is a test!")
    def update_metrics(self):
        self.needle_entry_dist = self._comm_channel.AVG_needle_entry_dist
        self.grasp_steps = self._comm_channel.grasp_steps
        #self.optimal_action_error = self._comm_channel.accumulated_optimal_action_error_norm
        #print(f'needle_entry_dist: {self.needle_entry_dist}')
        #print(f'grasp_steps: {self.grasp_steps}')

    def action_spec(self):
        return BoundedArraySpec(shape=self.action_space.shape,
                                dtype=np.float32,
                                minimum=self.action_space.low,
                                maximum=self.action_space.high,
                                name='action')
    def reset(self):
        print(f'Total reward: {self._total_reward}')
        print(f"Reset was called")
        print(f'Running in mode: {self.mode}')
        # Reset the simulation and get the initial observation
        self.set_target_entry(random.choice([2, 3]))
        #self.set_target_entry(3)
        self.steps = 0


        self.needle_entry_dist = 999
        #self.optimal_action_error = 0
        self.grasp_steps = 0
        self._total_reward = 0
        self._comm_channel.reset_world(self.done)
        self.done = False
        self._state = self._comm_channel.get_observation()
        return self._state
    def step(self, action):
        # Send the action to the simulator and get the new observation and reward
        px, py, pz, ox, oy, oz, ow, gripper_angle = action
        print(f'STEP {self.steps} Agent action: {action}')
        self.steps += 1
        for _ in range(self.repeat_actions):
            self._comm_channel.send_action([px, py, pz, ox, oy, oz, ow])
            # This ensures that the gripper can only be open or closed
            thres_gripper = np.pi/4 if gripper_angle >= 0.5 else 0
            self._comm_channel.move_jaw('psm2', thres_gripper)
            #self._comm_channel.move_jaw('psm2', gripper_angle)
        self._state = self._comm_channel.get_observation()

        state = self._state
        self.done = True if self.steps >= self.n_steps or self._comm_channel.done else False
        reward = self.get_reward(self.done)
        self._total_reward += reward
        info = {}

        if self._video:  # Add frame # TODO: Try and fix video framerate
            self._add_frame()

        # Create a video from frame buffer if done
        if self._video and self.done:
            self._create_video()

        # Update Metrics after every high level action
        if not self.done:
            self._comm_channel._update_metrics()
            self.update_metrics()


        # open the file in the write mode
        with open(self.csv_file, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            # Add the data in row
            row = [self.steps, reward, self.needle_entry_dist, self.grasp_steps]
            # write a row to the csv file
            writer.writerow(row)

        return state, reward, self.done, info

    def get_reward(self, done):
        #return 1.0 if done else 0
        return self._comm_channel.get_reward(done)

    def set_target_entry(self, entry_target_num):
        self._comm_channel.set_target_entry(entry_target_num)

    def init_arm(self, action):
        self.step(action)
        time.sleep(3)
    def get_O_action(self):
        return self._comm_channel.get_Oracle_action()

    def _add_frame(self):
        image = self._comm_channel.get_video_feed()
        self._frames.append(image)


    def set_mode_demo(self):
        self.mode = "DEMO"

    def _create_video(self):
        now = datetime.datetime.now()

        filename = f"{self.video_title}_{now.strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
        if self.mode == "DEMO":
            filename = f"{self.video_title}.mp4"


        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # choose a codec (e.g., H264, MP4V, etc.)
        out = cv2.VideoWriter(filename, fourcc, 25, (480, 270))  # output filename, codec, FPS, and resolution

        # Loop through the array of images and write each frame to the video
        for img in self._frames:
            out.write(img)

        # Release the VideoWriter and close the output file
        out.release()
        self._frames = [] # Reset frame buffer

        print(f"Video created: {filename}")


class AverageNeedleEntryDistance(py_metrics.StreamingMetric):
    def __init__(self, env, name='AverageNeedleEntryDistance', buffer_size=3, batch_size=None):
        """Creates an AverageReturnMetric."""
        self._env = env
        super(AverageNeedleEntryDistance, self).__init__(name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        pass

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
        trajectory: a tf_agents.trajectory.Trajectory.
        """
        lasts = trajectory.is_last()
        if np.any(lasts):
            is_last = np.where(lasts)
            goal_distance = np.asarray(self._env.needle_entry_dist, np.float32)

            if goal_distance.shape is ():  # pylint: disable=literal-comparison
                goal_distance = nest_utils.batch_nested_array(goal_distance)

            self.add_to_buffer(goal_distance[is_last])

class AverageGraspSteps(py_metrics.StreamingMetric):
    def __init__(self, env, name='AverageGraspSteps', buffer_size=3, batch_size=None):
        """Creates an AverageReturnMetric."""
        self._env = env
        super(AverageGraspSteps, self).__init__(name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        pass

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
        trajectory: a tf_agents.trajectory.Trajectory.
        """
        lasts = trajectory.is_last()
        if np.any(lasts):
            is_last = np.where(lasts)
            goal_distance = np.asarray(self._env.grasp_steps, np.float32)

            if goal_distance.shape is ():  # pylint: disable=literal-comparison
                goal_distance = nest_utils.batch_nested_array(goal_distance)

            self.add_to_buffer(goal_distance[is_last])




# Register environment, so it can be loaded via name
if 'SurgicalEnv-v2' in registration.registry.env_specs:
    del registration.registry.env_specs['SurgicalEnv-v2']
registration.register(id='SurgicalEnv-v2', entry_point=SurgicalEnv)

if __name__ == "__main__":
    from tf_agents.environments import suite_gym
    env = suite_gym.load('SurgicalEnv-v2')
    timestep = env.reset()

    # Get the action space
    action_spec = env.action_spec()

    # Generate a random action within the action space
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    timestep = env.step(action)  # TODO: Fix time_step error

    env.close()
