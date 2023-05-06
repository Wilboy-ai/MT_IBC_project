import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from tf_agents.networks import q_network
from tf_agents.networks import sequential
from tf_agents.networks import network
import tf_agents.networks


from tf_agents.agents import BehavioralCloningAgent
from eval import get_eval_actor
from load_data import create_dataset_fn, get_normalizers
from network import get_energy_model
from tf_agents.networks import sequential
import particle_env
from utils import get_sampling_spec, make_video
import mse_agent
import get_cloning_network


def training_step(bc_learner, fused_train_steps, train_step):
    print("Inside training_step")

    """Runs training step and saves the loss to tensorboard"""
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)

    print("bc_learner.run works")

    if reduced_loss_info:
        with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(True):
            tf.summary.scalar('reduced_loss', reduced_loss_info.loss, step=train_step)
    print("reduced_loss_info works")



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
    return eval_actor.metrics


def train():
    logging.set_verbosity(logging.INFO)

    tf.random.set_seed(0)
    root_dir = 'output/'
    dataset_path = 'data/2d_oracle_particle_*.tfrecord'
    network_width = 256
    batch_size = 512
    num_iterations = 2000
    video = True
    sequence_length = 2
    learning_rate = 1e-3,
    decay_steps = 100,

    # Load openai gym for evaluating the learned policy
    env_name = "Particle-v1"
    eval_env = suite_gym.load(env_name)
    eval_env = wrappers.HistoryWrapper(eval_env, history_length=2, tile_first_step_obs=True)

    # Get shape of observation and action space
    obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (spec_utils.get_tensor_specs(eval_env))

    #print(obs_tensor_spec)
    # OrderedDict([('pos_agent', BoundedTensorSpec(shape=(2, 2), dtype=tf.float32, name='observation/pos_agent', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32))), ('vel_agent', BoundedTensorSpec(shape=(2, 2), dtype=tf.float32, name='observation/vel_agent', minimum=array(-100., dtype=float32), maximum=array(100., dtype=float32))), ('pos_first_goal', BoundedTensorSpec(shape=(2, 2), dtype=tf.float32, name='observation/pos_first_goal', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32))), ('pos_second_goal', BoundedTensorSpec(shape=(2, 2), dtype=tf.float32, name='observation/pos_second_goal', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)))])

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
        action_sampling_spec = get_sampling_spec(action_tensor_spec,
                                                 min_actions=norm_info.min_actions,
                                                 max_actions=norm_info.max_actions,
                                                 act_norm_layer=norm_info.act_norm_layer,
                                                 uniform_boundary_buffer=0.05)

        print("##### action_sampling_spec #####")
        print(action_sampling_spec)


        # Create cloning network for BCAgent
        cloning_network = get_cloning_network.get_cloning_network(name="bc_model",
                                                           obs_tensor_spec=obs_tensor_spec,
                                                           action_tensor_spec=action_tensor_spec,
                                                           obs_norm_layer=norm_info.obs_norm_layer,
                                                           act_norm_layer=norm_info.act_norm_layer,
                                                           sequence_length=sequence_length,
                                                           act_denorm_layer=norm_info.act_denorm_layer)
        print("Get network works")

        # Define learning rate scheduler
        learning_rate_schedule = (
            tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=decay_steps, decay_rate=0.99))
        #learning_rate_schedule = WarmupSchedule(lr=learning_rate)

        print("learning_rate_schedule works")

        # Define optimizer for BC agent
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        print("optimizer works")

        #initilize BC agent
        agent = mse_agent.MseBehavioralCloningAgent(time_step_spec=time_step_tensor_spec,
                                                    action_spec=action_tensor_spec,
                                                    action_sampling_spec=action_sampling_spec,
                                                    obs_norm_layer=norm_info.obs_norm_layer,
                                                    act_norm_layer=norm_info.act_norm_layer,
                                                    act_denorm_layer=norm_info.act_denorm_layer,
                                                    cloning_network=cloning_network,
                                                    optimizer=optimizer,
                                                    train_step_counter=train_step)
        agent.initialize()

        print("initialize agent works")

        # Save model
        # Do i need this? :)
        saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
        extra_concrete_functions = []
        print("extra_concrete_functions works")

        '''
        try:
            BC_network_fn = mse_agent.generate_registration_functions(agent.policy, agent.cloning_network, strategy)
            extra_concrete_functions = [('cloning_network', BC_network_fn)]
        except ValueError:
            print('Unable to generate concrete functions. Skipping.')

        save_model_trigger = triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step, interval=1000,
                                                              extra_concrete_functions=extra_concrete_functions,
                                                              use_nest_path_signatures=False, save_greedy_policy=True)
                                                              '''
        save_model_trigger = triggers.PolicySavedModelTrigger(saved_model_dir, agent, train_step, interval=1000,
                                                              use_nest_path_signatures=False, save_greedy_policy=True)

        print("save_model_trigger works")

        # Create TFAgents driver for running the training process
        def dataset_fn():
            training_data = create_train_fns()
            return training_data

        '''bc_learner = learner.Learner(root_dir, train_step, agent, dataset_fn,
                                     triggers=[save_model_trigger,
                                               triggers.StepPerSecondLogTrigger(train_step, interval=100)],
                                     checkpoint_interval=5000, summary_interval=100, strategy=strategy,
                                     run_optimizer_variable_init=False)'''

        bc_learner = learner.Learner(root_dir, train_step, agent, dataset_fn,
                                     triggers=[triggers.StepPerSecondLogTrigger(train_step, interval=100)],
                                     checkpoint_interval=5000, summary_interval=100, strategy=strategy,
                                     run_optimizer_variable_init=False)
        print("bc_learner works")

        # Create TFAgents actor which applies the learned policy in the gym environment
        eval_actor, eval_success_metric = get_eval_actor(agent, eval_env, train_step, root_dir, strategy,
                                                         env_name.replace('/', '_'))
        # get_eval_loss = tf.function(agent.get_eval_loss)

        print("get_eval_actor works")

        aggregated_summary_dir = os.path.join(root_dir, 'eval')
        summary_writer = tf.summary.create_file_writer(aggregated_summary_dir, flush_millis=10000)

        print("aggregated_summary_dir works")

    training_step(bc_learner, 50, train_step)
    print("training_step works")
    evaluation_step(eval_env, eval_actor, eval_episodes=20)
    print("evaluation_step works")
    eval_env.seed(42)
    make_video(agent, eval_env, root_dir, step=train_step.numpy(), strategy=strategy)
    print("make_video works")

    # Main train loop
    while train_step.numpy() < num_iterations:
        training_step(bc_learner, 50, train_step)

        # Evaluate policy
        if train_step.numpy() % 250 == 0:
            all_metrics = []

            metrics = evaluation_step(eval_env, eval_actor, eval_episodes=20)
            all_metrics.append(metrics)

            # Generate video of learned policy
            if video:
                eval_env.seed(42)
                make_video(agent, eval_env, root_dir, step=train_step.numpy(), strategy=strategy)

            metric_results = collections.defaultdict(list)
            for env_metrics in all_metrics:
                for metric in env_metrics:
                    metric_results[metric.name].append(metric.result())

            with summary_writer.as_default(), common.soft_device_placement(), tf.summary.record_if(lambda: True):
                for key, value in metric_results.items():
                    tf.summary.scalar(name=os.path.join('AggregatedMetrics/', key), data=sum(value) / len(value),
                                      step=train_step)

    summary_writer.flush()


def main(_):
    tf.config.experimental_run_functions_eagerly(False)
    train()


if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))
