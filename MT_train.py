import random
from os import environ
environ['XLA_FLAGS']="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

import tensorflow as tf
import functools
from absl import app
from absl import logging
import os
import collections

from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.utils import common

from agent import ImplicitBCAgent, generate_registration_functions
from eval import get_eval_actor
from load_data import create_dataset_fn, get_normalizers
from network import get_energy_model
from utils import get_sampling_spec, make_video

from MT_RosAmbfCommChannel import RosAmbfCommChannel
import MT_Simple_AccelNet_env
import numpy as np
import pickle
import dill
import wandb
import datetime
import matplotlib
import time
# TODO: Clean up imports

matplotlib.use('Agg') # memory issue fix? # TODO: Remove this? I don't think it is relevant anymore...

# Caps the GPU memory to 80% (18GB)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #tf.config.experimental.set_virtual_device_configuration(
        #    gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    except RuntimeError as e:
        print(e)


def training_step(bc_learner, fused_train_steps, train_step):
    """Runs training step and saves the loss to tensorboard"""
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)

    # if reduced_loss_info:
    #     with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
    #         tf.summary.scalar('reduced_loss', reduced_loss_info.loss, step=train_step)

    return reduced_loss_info

def evaluation_step(eval_env, eval_actor, eval_episodes):
    """Runs evaluation routine in gym and returns metrics."""
    logging.info('Evaluation policy.')
    custom_eval_seeds = [67, 29, 56, 182, 84, 94654, 549, 8423, 54, 80]

    with tf.name_scope('eval'):
        for eval_seed in range(eval_episodes):
            eval_env.seed(custom_eval_seeds[eval_seed])
            eval_actor.reset()
            eval_actor.run()

        eval_actor.log_metrics()
        eval_actor.write_metric_summaries()
    return eval_actor.metrics


def train(iteration_number):
    logging.set_verbosity(logging.INFO)

    tf.random.set_seed(0)
    root_dir = f'IBC_RUNS/AE-{iteration_number}/'
    dataset_path = 'Trainings_Data/AUTOENCODER_DATASET/suture_throw_demo_*.tfrecord'

    # This is logged by wandb, can help with bookkeeping
    print(f'Run name: {root_dir}')
    print(f'Using data from: {dataset_path}')

    # Make root folder
    # Make root folder if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # make video folder
    save_video_path = f'{root_dir}Videos'
    # Make root folder if it doesn't exist
    if not os.path.exists(save_video_path):
        os.makedirs(save_video_path)

    print(f'Running: {root_dir}')
    # Load in hyper-parameters from wandb config
    network_width = wandb.config.network_width
    network_depth = wandb.config.network_depth
    batch_size = wandb.config.batch_size
    num_iterations = wandb.config.num_iterations
    num_counter_examples = wandb.config.num_counter_examples
    learning_rate = wandb.config.learning_rate
    num_action_samples = wandb.config.num_action_samples
    langevin_iteration = wandb.config.langevin_iteration
    langevin_stepsize = wandb.config.langevin_stepsize
    eval_interval = wandb.config.eval_interval
    eval_episodes = wandb.config.eval_episodes
    decay_steps = wandb.config.decay_steps
    decay_rate = wandb.config.decay_rate

    # Load openai gym for evaluating the learned policy
    env_name = "SurgicalEnv-v2"
    # Retry Loading environment logic
    # It doesn't always connect to the launch_crtk_interface, so you have to retry, idk how to fix it..
    for i in range(0, 5):
        try:
            eval_env = suite_gym.load(env_name)
            break
        except Exception:
            print(f'Failed to load SurgicalEnv-v2, will try again ({i}/5)')
            time.sleep(1)
            continue

    eval_env = wrappers.HistoryWrapper(eval_env, history_length=2, tile_first_step_obs=True)
    # TODO: What should these be history_length=2, tile_first_step_obs=True




    #Get shape of observation and action space
    obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (spec_utils.get_tensor_specs(eval_env))

    # Get dataset statistics for data normalization
    create_train_unnormalized = create_dataset_fn(dataset_path, batch_size)

    train_data = create_train_unnormalized()

    (norm_info, norm_train_data_fn) = get_normalizers(train_data)

    # Creates tf.distribute.MirroredStrategy to enable computation on multiple GPUs
    strategy = strategy_utils.get_strategy(tpu=None, use_gpu=True)
    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

    # Create dataloader with normalization layers
    create_train_fns = create_dataset_fn(dataset_path, per_replica_batch_size, norm_train_data_fn)

    with strategy.scope():
        # Train step counter for MirroredStrategy
        train_step = train_utils.create_train_step()

        # Calculate minimum and maximum value of action space for sampling counter examples
        action_sampling_spec = get_sampling_spec(action_tensor_spec, min_actions=norm_info.min_actions,
                                                 max_actions=norm_info.max_actions,
                                                 act_norm_layer=norm_info.act_norm_layer,
                                                 uniform_boundary_buffer=0.05)

        # Create MLP (= probabilistically modelled energy function)
        energy_model = get_energy_model(obs_tensor_spec, action_tensor_spec, network_depth, network_width)

        # Wrapper which contains the training process
        agent = ImplicitBCAgent(time_step_spec=time_step_tensor_spec, action_spec=action_tensor_spec,
                                action_sampling_spec=action_sampling_spec, obs_norm_layer=norm_info.obs_norm_layer,
                                act_norm_layer=norm_info.act_norm_layer, act_denorm_layer=norm_info.act_denorm_layer,
                                cloning_network=energy_model, num_counter_examples=num_counter_examples,
                                train_step_counter=train_step, learning_rate=learning_rate,
                                num_action_samples=num_action_samples,
                                langevin_iteration=langevin_iteration,
                                langevin_stepsize=langevin_stepsize,
                                decay_steps=decay_steps,
                                decay_rate=decay_rate
                                )
        agent.initialize()

        # Save model
        saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
        extra_concrete_functions = []
        try:
            energy_network_fn = generate_registration_functions(agent.policy, agent.cloning_network, strategy)
            extra_concrete_functions = [('cloning_network', energy_network_fn)]
        except ValueError:
            print('Unable to generate concrete functions. Skipping.')

        save_model_trigger = triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step, interval=1000,
                                                              extra_concrete_functions=extra_concrete_functions,
                                                              use_nest_path_signatures=False, save_greedy_policy=True)

        # Create TFAgents driver for running the training process32
        def dataset_fn():
            training_data = create_train_fns()
            return training_data
        bc_learner = learner.Learner(root_dir, train_step, agent, dataset_fn,
                                     triggers=[save_model_trigger,
                                               triggers.StepPerSecondLogTrigger(train_step, interval=100)],
                                     checkpoint_interval=5000, summary_interval=100, strategy=strategy,
                                     run_optimizer_variable_init=False)
        # Create TFAgents actor which applies the learned policy in the gym environment
        eval_actor, eval_success_metric = get_eval_actor(agent, eval_env, train_step, root_dir, strategy,
                                                         env_name.replace('/', '_'))

        aggregated_summary_dir = os.path.join(root_dir, 'eval')
        summary_writer = tf.summary.create_file_writer(aggregated_summary_dir, flush_millis=10000)

    # Container for wandb metrics
    wandb_all_metrics = {}

    trainings_metrics = training_step(bc_learner, 50, train_step)
    eval_env.set_video_title(f'{save_video_path}/Suture_eval_50')
    eval_env.set_csv_file(f'{root_dir}Suture_eval_50.csv')
    eval_metrics = evaluation_step(eval_env, eval_actor, eval_episodes=3) # Only run the start bench 3 times

    def update_wandb_all_metrics(_trainings_metrics, _eval_metrics):
        for name, metric in _trainings_metrics.extra.items():
            wandb_all_metrics[name] = metric
        wandb_all_metrics['loss'] = _trainings_metrics.loss

        for metric in _eval_metrics:
            name = metric.name
            result = metric.result()
            wandb_all_metrics[name] = result

    update_wandb_all_metrics(trainings_metrics, eval_metrics)

    best_score = wandb_all_metrics["AverageReturn"]

    # Main train loop
    print(f'Step starting at: {train_step.numpy()}')
    while train_step.numpy() < num_iterations:
        print(f'At training step: {train_step.numpy()}/{num_iterations}')

        trainings_metrics = training_step(bc_learner, 50, train_step)

        update_wandb_all_metrics(trainings_metrics, eval_metrics)
        wandb.log(wandb_all_metrics)

        # Early stop function
        newest_score = wandb_all_metrics["AverageReturn"]
        print(f'best_score; {best_score} vs. {newest_score}')
        if best_score < newest_score:
            best_score = newest_score
        #if train_step.numpy() >= 5000 and best_score <= 1:
        #    print(f'Early stopping!')
        #    break

        # Evaluate policy
        if train_step.numpy() % eval_interval == 0:
            print(f'Training at step: {train_step.numpy()}/{num_iterations}')

            # Generate video of learned policy
            eval_env.set_video_title(f'{save_video_path}/Suture_eval_{train_step.numpy()}')
            eval_env.set_csv_file(f'{root_dir}Suture_eval_{train_step.numpy()}.csv')
            eval_metrics = evaluation_step(eval_env, eval_actor, eval_episodes=eval_episodes)



    summary_writer.flush()
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

def main(_):
    for iteration_number in range(0, 1):
        # wandb project name
        project_name = "MT_AUTOENCODER"

        # Optimal Configurations
        # num_iterations = 15000
        # eval_interval = 2000
        # eval_episodes = 10
        # network_width = 190
        # network_depth = 2
        # batch_size = 64
        # num_counter_examples = 4
        # learning_rate = 1e-2
        # num_action_samples = 1024
        # langevin_iteration = 60
        # langevin_stepsize = 1e-11
        # decay_steps = 80
        # decay_rate = 0.99

        # Default Configurations
        num_iterations = 25000
        eval_interval = 2000
        eval_episodes = 10
        network_width = 256 #256 #128
        network_depth = 2 #2
        batch_size = 256# 128
        num_counter_examples = 4
        learning_rate = 0.01
        num_action_samples = 128 #512
        langevin_iteration = 126 #100
        langevin_stepsize = 1.e-3 #1.e-5
        decay_steps = 125 #100
        decay_rate = 0.99

        # Default Configurations sweep
        # num_iterations = 15000
        # eval_interval = 2000
        # eval_episodes = 10
        # network_width = random.choice([64, 128, 258])
        # network_depth = random.choice([2, 4])
        # batch_size = random.choice([16, 32, 64, 128, 256, 512])
        # num_counter_examples = random.choice([4, 8, 16])
        # learning_rate = random.choice([1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5])
        # num_action_samples = random.choice([128, 256, 512, 1024])
        # langevin_iteration = random.choice([50, 100, 250, 500])
        # langevin_stepsize = random.choice([1.e-4, 1.e-5, 1.e-6, 1.e-7])
        # decay_steps = random.choice([50, 100, 250, 500])
        # decay_rate = random.choice([0.8, 0.95, 0.99])

        # Configurations tuning sweep
        # num_iterations = 15000  # random.choice([50000])
        # eval_interval = 2000
        # eval_episodes = 10
        # network_width = 190 + (10 * random.randint(-10, 10))               #random.choice([128, 200, 256, 300, 512])
        # network_depth = 2 + (2 * random.randint(1, 5))
        # batch_size = 64 + (10 * random.randint(-5, 5))                     #random.choice([64, 80, 128, 150, 256])
        # num_counter_examples = 4  + random.randint(0, 5)           #random.choice([4, 6, 8, 10])
        # learning_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])  # 1e-2             #random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        # num_action_samples = 1024 + (10 * random.randint(-100, 100))             #random.choice([64, 128, 256, 512, 1024])
        # langevin_iteration = 60 + (10 * random.randint(-5, 5))            #random.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        # langevin_stepsize = random.choice([1e-9, 1e-10, 1e-11, 1e-12, 1e-13])  #1e-11           #random.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])            # Final langevin stepsizes
        # decay_steps = 80 + (10 * random.randint(-5, 500))                    #random.choice([100, 200, 300, 400, 500])
        # decay_rate = random.uniform(0.8, 0.99)  #              #random.choice([0.8, 0.95, 0.99]) #0.99

        # # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # track hyperparameters and run metadata
            config={
                'num_iterations': num_iterations,
                'eval_interval': eval_interval,
                'eval_episodes': eval_episodes,
                'network_width': network_width,
                'network_depth': network_depth,
                'batch_size': batch_size,
                'num_counter_examples': num_counter_examples,
                'learning_rate': learning_rate,
                'num_action_samples': num_action_samples,
                'langevin_iteration': langevin_iteration,
                'langevin_stepsize': langevin_stepsize,
                'decay_steps': decay_steps,
                'decay_rate': decay_rate,
            }
        )

        tf.config.experimental_run_functions_eagerly(False)
        train(iteration_number)

if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))

