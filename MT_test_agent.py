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

import wandb
import datetime
import matplotlib
matplotlib.use('Agg') # memory issue fix?



global global_run_count
global_run_count = 2
print(f'global_run_count: {global_run_count}')
project_name = "PixelEBM_MultiEntry"


def training_step(bc_learner, fused_train_steps, train_step):
    """Runs training step and saves the loss to tensorboard"""
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
    #print(f'reduced_loss_info: {dir(reduced_loss_info)}')

    if reduced_loss_info:
        with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
            tf.summary.scalar('reduced_loss', reduced_loss_info.loss, step=train_step)

    return reduced_loss_info

def evaluation_step(eval_env, eval_actor, eval_episodes):
    """Runs evaluation routine in gym and returns metrics."""
    logging.info('Evaluation policy.')
    with tf.name_scope('eval'):
        for eval_seed in range(eval_episodes):
            eval_env.seed(eval_seed)
            eval_actor.reset()
            eval_actor.run()

        eval_actor.log_metrics()
        eval_actor.write_metric_summaries()
        #print(f'eval_actor.metrics: {eval_actor.metrics}')
    return eval_actor.metrics


def train():
    logging.set_verbosity(logging.INFO)
    #wandb.init(project=project_name)

    tf.random.set_seed(0)
    root_dir = f'visu_mono_PixelEBM_{global_run_count}/'
    dataset_path = 'ambf_data/suture_throw_demo_*.tfrecord'
    run_prefix = f'visu_mono_PixelEBM_{global_run_count}'

    network_width = 50000
    batch_size = 128
    num_iterations = 64
    num_counter_examples = 4
    learning_rate = 1e-3

    # Load openai gym for evaluating the learned policy
    env_name = "SurgicalEnv-v2"
    eval_env = suite_gym.load(env_name)
    eval_env = wrappers.HistoryWrapper(eval_env, history_length=2, tile_first_step_obs=True) #Not sure i we need this?
    # TODO: What should these be history_length=2, tile_first_step_obs=True

    print("Observation Space:", eval_env.observation_space)
    print("Action Space:", eval_env.action_space)

    # #Get shape of observation and action space
    obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (spec_utils.get_tensor_specs(eval_env))
    #
    # Get dataset statistics for data normalization
    create_train_unnormalized = create_dataset_fn(dataset_path, batch_size)

    train_data = create_train_unnormalized()

    (norm_info, norm_train_data_fn) = get_normalizers(train_data)
    #
    # # Creates tf.distribute.MirroredStrategy to enable computation on multiple GPUs
    strategy = strategy_utils.get_strategy(tpu=None, use_gpu=True)
    #per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
    #
    # # Create dataloader with normalization layers
    # create_train_fns = create_dataset_fn(dataset_path, per_replica_batch_size, norm_train_data_fn)

    with strategy.scope():
        # Train step counter for MirroredStrategy
        train_step = train_utils.create_train_step()

        # Calculate minimum and maximum value of action space for sampling counter examples
        action_sampling_spec = get_sampling_spec(action_tensor_spec, min_actions=norm_info.min_actions,
                                                 max_actions=norm_info.max_actions,
                                                 act_norm_layer=norm_info.act_norm_layer,
                                                 uniform_boundary_buffer=0.05)

        # Create MLP (= probabilistically modelled energy function)
        energy_model = get_energy_model(obs_tensor_spec, action_tensor_spec, network_width)

        # Wrapper which contains the training process
        agent = ImplicitBCAgent(time_step_spec=time_step_tensor_spec, action_spec=action_tensor_spec,
                                action_sampling_spec=action_sampling_spec, obs_norm_layer=norm_info.obs_norm_layer,
                                act_norm_layer=norm_info.act_norm_layer, act_denorm_layer=norm_info.act_denorm_layer,
                                cloning_network=energy_model, num_counter_examples=num_counter_examples, train_step_counter=train_step, learning_rate=learning_rate)
        agent.initialize()

        # # Save model
        # saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
        # extra_concrete_functions = []
        # try:
        #     energy_network_fn = generate_registration_functions(agent.policy, agent.cloning_network, strategy)
        #     extra_concrete_functions = [('cloning_network', energy_network_fn)]
        # except ValueError:
        #     print('Unable to generate concrete functions. Skipping.')
        #
        # save_model_trigger = triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step, interval=1000,
        #                                                       extra_concrete_functions=extra_concrete_functions,
        #                                                       use_nest_path_signatures=False, save_greedy_policy=True)

        # Create TFAgents driver for running the training process
        # def dataset_fn():
        #     training_data = create_train_fns()
        #     return training_data
        # bc_learner = learner.Learner(root_dir, train_step, agent, dataset_fn,
        #                              triggers=[save_model_trigger,
        #                                        triggers.StepPerSecondLogTrigger(train_step, interval=100)],
        #                              checkpoint_interval=5000, summary_interval=100, strategy=strategy,
        #                              run_optimizer_variable_init=False)
        # # Create TFAgents actor which applies the learned policy in the gym environment
        eval_actor, eval_success_metric = get_eval_actor(agent, eval_env, train_step, root_dir, strategy,
                                                         env_name.replace('/', '_'))

        #get_eval_loss = tf.function(agent.get_eval_loss())
        #print(f'Got pass get_eval_actor, {get_eval_loss}')
        #print(f'get_eval_loss: {agent.get_eval_loss()}')

        aggregated_summary_dir = os.path.join(root_dir, 'eval')
        summary_writer = tf.summary.create_file_writer(aggregated_summary_dir, flush_millis=10000)

    #training_step(bc_learner, 50, train_step)
    eval_env.set_video_title(f'{run_prefix}_50')
    evaluation_step(eval_env, eval_actor, eval_episodes=1)

    best_get_needle_entry_dist = eval_env.min_needle_entry_dist
    print(f'best_get_needle_entry_dist:  {best_get_needle_entry_dist}')

    # min_loss = 99999
    # early_stop_thres = 50
    # early_stop_count = 0
    #
    # # Main train loop
    # while train_step.numpy() < num_iterations:
    #     loss = training_step(bc_learner, 50, train_step)
    #     # log the accuracy and loss to wandb
    #
    #     wandb.log({'loss': loss.loss})
    #     wandb.log({'needle_entry_dist': best_get_needle_entry_dist})
    #
    #     # Early stopping algorithm
    #     print(f'min_loss: {min_loss} vs {loss.loss}, early_stop_count: {early_stop_count}/{early_stop_thres}')
    #     if loss.loss < min_loss:
    #         min_loss = loss.loss
    #         early_stop_count = 0
    #     else:
    #         early_stop_count += 1
    #
    #     #if early_stop_count >= early_stop_thres:
    #     #    break
    #
    #     # Evaluate policy
    #     if train_step.numpy() % 5000 == 0:
    #         print(f'Training at step: {train_step.numpy()}/{num_iterations}')
    #         all_metrics = []
    #
    #         # Generate video of learned policy
    #         #for i in range(0, 5):
    #         eval_env.set_video_title(f'{run_prefix}_{train_step.numpy()}')
    #         metrics = evaluation_step(eval_env, eval_actor, eval_episodes=1)
    #         #new_get_needle_entry_dist = metrics[0] #TODO: Make an average function for this metric
    #         #all_metrics.append(metrics)
    #         new_get_needle_entry_dist = eval_env.min_needle_entry_dist
    #         if new_get_needle_entry_dist < best_get_needle_entry_dist:
    #             best_get_needle_entry_dist = new_get_needle_entry_dist
    #
    #         metric_results = collections.defaultdict(list)
    #         for env_metrics in all_metrics:
    #             for metric in env_metrics:
    #                 metric_results[metric.name].append(metric.result())
    #
    #         with summary_writer.as_default(), common.soft_device_placement(), tf.summary.record_if(lambda: True):
    #             for key, value in metric_results.items():
    #                 tf.summary.scalar(name=os.path.join('AggregatedMetrics/', key), data=sum(value)/len(value),
    #                                   step=train_step)
    #
    #
    #
    #
    # summary_writer.flush()
    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()


def main(_):
    # num_iterations = 50000 #random.choice([50000])
    # network_width = 128 #random.choice([128, 256, 512])
    # batch_size = 64 #random.choice([64, 128, 256])
    # num_counter_examples = 4 #random.choice([4, 6, 8, 10])
    # learning_rate = 1e-3 #random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    #
    # # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=project_name,
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         'num_iterations': num_iterations,
    #         'network_width': network_width,
    #         'batch_size': batch_size,
    #         'num_counter_examples': num_counter_examples,
    #         'learning_rate': learning_rate,
    #     }
    # )

    tf.config.experimental_run_functions_eagerly(False)
    train()
#    global_run_count += 1

# def dummy_funciton():
#     wandb.init(project=project_name)
#     network_width = wandb.config.network_width
#     batch_size = wandb.config.batch_size
#     num_iterations = wandb.config.num_iterations
#     num_counter_examples = wandb.config.num_counter_examples
#     learning_rate = wandb.config.learning_rate
#
#     for i in range(0, 10):
#         wandb.log({'loss': random.randint(0,10)})

#def sweep_train_test():
    #multiprocessing.handle_main(functools.partial(app.run, main))
    #dummy_funciton() # Function to test the setup of
    #sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    #wandb.agent(sweep_id, function=main, count=1)

# Start the sweep
#sweep_id = wandb.sweep(sweep_configuration, project=project_name)
#wandb.agent(sweep_id, function=sweep_train_test, count=1)
#multiprocessing.handle_main(functools.partial(app.run, main))

if __name__ == "__main__":
    # create the alarm clock.
    #alarm = datetime.time(0, 0, 0)  # Hour, minute and second you want.
    #while alarm < datetime.datetime.now().time():
    multiprocessing.handle_main(functools.partial(app.run, main))

    #sweep_train_test()
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    #wandb.agent(sweep_id, function=sweep_train_test, count=1)
