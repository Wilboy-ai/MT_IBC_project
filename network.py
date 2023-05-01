import tensorflow as tf
from tf_agents.utils import nest_utils
from tf_agents.networks import network
from MT_layers import get_conv_maxpool, DenseResnetValue
from tf_agents.policies import py_tf_eager_policy
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file



class ResNetPreActivationLayer(tf.keras.layers.Layer):
    """ResNet layer, improved 'pre-activation' version."""
    def __init__(self, hidden_sizes, rate, kernel_initializer, bias_initializer, **kwargs):

        super(ResNetPreActivationLayer, self).__init__(**kwargs)

        # ResNet wants layers to be even numbers, but remember there will be an additional
        # layer just to project to the first hidden size.
        assert len(hidden_sizes) % 2 == 0
        self._projection_layer = tf.keras.layers.Dense(hidden_sizes[0], activation=None,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer)
        self._weight_layers = []
        self._norm_layers = []
        self._activation_layers = []
        self._dropouts = []

        self._weight_layers_2 = []
        self._norm_layers_2 = []
        self._activation_layers_2 = []
        self._dropouts_2 = []

        def create_dense_layer(width):
            return tf.keras.layers.Dense(width, activation=None, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer)

        # Step every other
        for layer in range(0, len(hidden_sizes), 2):
            self._weight_layers.append(create_dense_layer(hidden_sizes[layer]))
            self._activation_layers.append(tf.keras.layers.ReLU())
            self._dropouts.append(tf.keras.layers.Dropout(rate))

            self._weight_layers_2.append(create_dense_layer(hidden_sizes[layer+1]))
            self._activation_layers_2.append(tf.keras.layers.ReLU())
            self._dropouts_2.append(tf.keras.layers.Dropout(rate))

    def call(self, x, training):
        x = self._projection_layer(x)

        # Do forward pass through resnet layers.
        for layer in range(len(self._weight_layers)):
            x_start_block = tf.identity(x)
            x = self._activation_layers[layer](x, training=training)
            x = self._dropouts[layer](x, training=training)
            x = self._weight_layers[layer](x, training=training)

            x = self._activation_layers_2[layer](x, training=training)
            x = self._dropouts_2[layer](x, training=training)
            x = self._weight_layers_2[layer](x, training=training)
            x = x_start_block + x
        return x


class MLPEBM(network.Network):
    """MLP with ResNetPreActivation layers and Dense layer."""
    def __init__(self, obs_spec, action_spec, depth=2, width=256, name='MLPEBM'):
        super(MLPEBM, self).__init__(input_tensor_spec=obs_spec, state_spec=(), name=name)

        # Define MLP.
        hidden_sizes = [width for _ in range(depth)]
        self._mlp = ResNetPreActivationLayer(hidden_sizes, 0.0, 'normal', 'normal')

        # Define projection to energy.
        self._project_energy = tf.keras.layers.Dense(action_spec.shape[-1], kernel_initializer='normal',
                                                     bias_initializer='normal')

    def call(self, inputs, training, step_type=(), network_state=()):
        # obs: dict of named obs_spec.
        # act:   [B x act_spec]
        obs, act = inputs

        # Combine dict of observations to concatenated tensor. [B x T x obs_spec]
        obs = tf.concat(tf.nest.flatten(obs), axis=-1)

        # Flatten obs across time: [B x T * obs_spec]
        batch_size = tf.shape(obs)[0]
        obs = tf.reshape(obs, [batch_size, -1])

        # Concat [obs, act].
        x = tf.concat([obs, act], -1)

        # Forward mlp.
        x = self._mlp(x, training=training)

        # Project to energy.
        x = self._project_energy(x, training=training)

        # Squeeze extra dim.
        x = tf.squeeze(x, axis=-1)

        return x, network_state




"""Shared utilities for image preprocessing."""
def stack_images_channelwise(obs, batch_size):
  # Use static shapes for hist, width, height, and channels since TPUs prefer
  # static shapes for some image ops. The batch size passed in may still be
  # dynamic.
  nhist = obs.get_shape()[1]
  nw = obs.get_shape()[2]
  nh = obs.get_shape()[3]
  nc = obs.get_shape()[4]
  obs = tf.reshape(obs, tf.concat([[batch_size], [nw, nh, nc * nhist]], axis=0))
  return obs


def preprocess(images, target_height, target_width):
  """Converts to [0,1], stacks, resizes."""
  # Scale to [0, 1].
  images = tf.image.convert_image_dtype(images, dtype=tf.float32)

  # Stack images channel-wise.
  batch_size = tf.shape(images)[0]
  images = stack_images_channelwise(images, batch_size)

  # Resize to target height and width.
  images = tf.image.resize(images, [target_height, target_width])
  return images




def get_encoder_network(encoder_network, target_height, target_width, channels):
    return get_conv_maxpool(target_height, target_width, channels)


def get_value_network(value_network):
    return DenseResnetValue()


class PixelEBM(network.Network):
  """Late fusion PixelEBM."""
  def __init__(self,
               obs_spec,
               action_spec,
               encoder_network = 'ConvMaxpoolEncoder',
               value_network = 'DenseResnetValue',
               target_height=68,
               target_width=120,
               name='PixelEBM'):
    super(PixelEBM, self).__init__(
        input_tensor_spec=(obs_spec, action_spec),
        state_spec=(),
        name=name,
    )
    sequence_length = obs_spec['image'].shape[0]
    # We stack all images and coord-conv.
    num_channels = (3 * sequence_length)
    self._encoder = get_encoder_network(encoder_network,
                                        target_height,
                                        target_width,
                                        num_channels)
    self.target_height = target_height
    self.target_width = target_width
    self._value = get_value_network(value_network)

    rgb_shape = obs_spec['image'].shape
    print(f'obs_spec: {obs_spec}')
    print(f'rgb_shape: {rgb_shape}')
    self._static_height = rgb_shape[1]
    self._static_width = rgb_shape[2]
    self._static_channels = rgb_shape[3]

  def encode(self, obs, training):
    """Embeds images."""
    images = obs['image']

    # Ensure shape with static shapes from spec since shape information may
    # be lost in the data pipeline. ResizeBilinear is not supported with
    # dynamic shapes on TPU.
    # First 2 dims are batch size, seq len.
    images = tf.ensure_shape(images, [
        None, None, self._static_height, self._static_width,
        self._static_channels
    ])

    images = preprocess(images,
                         target_height=self.target_height,
                         target_width=self.target_width)
    observation_encoding = self._encoder(images, training=training)
    return observation_encoding
  def call(
      self,
      inputs,
      training,
      step_type=(),
      network_state=((), (), ()),
      observation_encoding=None):
    obs, act = inputs

    # If we pass in observation_encoding, we are doing late fusion.
    if observation_encoding is None:
      # Otherwise embed for the first time.
      observation_encoding = self.encode(obs, training)
      batch_size = tf.shape(obs['image'])[0]
      num_samples = tf.shape(act)[0] // batch_size
      observation_encoding = nest_utils.tile_batch(
          observation_encoding, num_samples)

    # Concat [obs, act].
    x = tf.concat([observation_encoding, act], -1)

    # Forward value network.
    x = self._value(x, training=training)

    # Squeeze extra dim.
    x = tf.squeeze(x, axis=-1)

    return x, network_state

def make_mlp_ebm(obs_spec, action_spec, depth=2, width=256, name='MLPEBM'):
    """MLP with ResNetPreActivation layers and Dense layer."""
    inputs = []
    for spec in tf.nest.flatten(obs_spec):
        inputs.append(tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype))

    input_act = tf.keras.layers.Input(shape=action_spec.shape, dtype=action_spec.dtype)
    inputs.append(input_act)

    # Combine dict of observations to concatenated tensor.
    x = tf.keras.layers.Concatenate(axis=-1)(inputs)

    # Flatten obs across time.
    x = tf.keras.layers.Flatten()(x)

    # Concat [obs, act].
    x = tf.keras.layers.Concatenate(axis=-1)([x, input_act])

    # Define MLP.
    hidden_sizes = [width for _ in range(depth)]
    for hidden_size in hidden_sizes:
        x = tf.keras.layers.Dense(hidden_size)(x)
        x = tf.keras.layers.ReLU()(x)

    # Define projection to energy.
    x = tf.keras.layers.Dense(action_spec.shape[-1], kernel_initializer='normal',
                                                     bias_initializer='normal')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    return model


# def create_model(input_shape):
#     model = Sequential()
#     model.add(Flatten(input_shape=input_shape[0]))  # flatten the first input tensor
#     model.add(Dense(256, activation='relu'))
#
#     # define the hidden layers
#     for i in range(2):
#         model.add(Dense(256, activation='relu'))
#
#     # define the concatenation layer
#     model.add(Concatenate())
#
#     # define the final dense layer to output the energy
#     model.add(Dense(1))
#
#     return model



def get_energy_model(obs_tensor_spec, action_tensor_spec, network_width):
    """Create MLP for simple features."""
    energy_model = MLPEBM(obs_spec=(obs_tensor_spec, action_tensor_spec), action_spec=tf.TensorSpec([1]),
                          width=network_width)

    #energy_model = make_mlp_ebm(obs_spec=(obs_tensor_spec, action_tensor_spec), action_spec=tf.TensorSpec([1]),
    #                      width=network_width)

    #energy_model.create_variables()

    #print(f'energy_model: {type(energy_model)}, {energy_model}')

    # # Display value before checkpoint restore
    # for layer in energy_model.trainable_variables:
    #     print(f'energy_model weights, {layer}')
    #     break
    #
    # tf.saved_model.save(obj=energy_model, export_dir="test/")
    # energy_model_restored = tf.saved_model.load('test/')
    # ibc_energy_model_restored = tf.saved_model.load('Sim_states_reduced_domain_5/policies/policy')
    #
    # # Display value before checkpoint restore
    # for layer in energy_model_restored.trainable_variables:
    #     print(f'energy_model_restored weights, {layer}')
    #     break
    #
    # for x in dir(energy_model):
    #     print(f'Checking {x}')
    #     if x not in dir(energy_model_restored):
    #         print(f"energy_model_restored: {x} is missing")
    #     if x not in dir(ibc_energy_model_restored):
    #         print(f"ibc_energy_model_restored: {x} is missing")
    #
    # energy_model = energy_model_restored
    #
    # print(f'energy_model_restored {type(energy_model_restored)} vs. {type(energy_model)}')
    # print(f'{dir(energy_model_restored)}')
    # print(f'{dir(energy_model)}')




    #print(f'_mlp: {energy_model._mlp}')

    # load_saved_model = False     # Toggle for debugging
    # if load_saved_model:
    #     print(f'Loading in saved energy model')
    #     # Create the weights of the new energy_model. To prevent errors of non-initiated weights
    #
    #     #Display value before checkpoint restore
    #     for layer in energy_model.trainable_variables:
    #         print(f'energy_model weights before checkpoint assignment, {layer}')
    #         break
    #
    #     # # Load a saved model
    #     saved_model_dir = "Sim_states_reduced_domain_5/policies/policy"
    #     model = tf.saved_model.load(saved_model_dir)
    #
    #
    #     # Create a checkpoint object for the policy
    #     #checkpoint = tf.train.Checkpoint(policy=energy_model)
    #     #checkpoint = tf.train.Checkpoint(model=model)
    #
    #     # Specify the checkpoint path
    #     #checkpoint_path = 'Sim_states_reduced_domain_5/policies/checkpoints/'
    #     #latest_checkpoint_path = 'Sim_states_reduced_domain_5/policies/checkpoints/policy_checkpoint_0000050000/variables/variables'
    #     #latest_checkpoint_path = 'Sim_states_reduced_domain_5/policies/checkpoints/policy_checkpoint_0000050000/variables/variables'
    #     #latest_checkpoint_path = 'Sim_states_reduced_domain_5/train/checkpoints/ckpt-50000'
    #     #print(f'latest_checkpoint path: {tf.train.latest_checkpoint(checkpoint_path)} vs. {checkpoint_path}')
    #
    #     #print('energy_model layers:')
    #     #energy_model.summary()
    #
    #     #print('print_tensors_in_checkpoint_file:')
    #     #print_tensors_in_checkpoint_file(file_name=latest_checkpoint_path, all_tensors=False, tensor_name='')
    #
    #     # # Get the variable names from the loaded checkpoint
    #     # variable_names = [v.name for v in tf.train.list_variables(latest_checkpoint_path + ".index")]
    #     #
    #     # # Match the variable names to the corresponding layers in your model
    #     # for variable_name in variable_names:
    #     #     print(f'variable_name: {variable_name}')
    #     #     if 'model_variables' in variable_name:
    #     #         layer_name = variable_name.split('/')[1]
    #     #         layer = getattr(energy_model, layer_name)
    #     #         var = tf.train.load_variable(latest_checkpoint_path, variable_name)
    #     #         layer.set_weights([var])
    #     #         print(f'{layer_name} was set!')
    #
    #     # Restore the checkpoint
    #     #status = checkpoint.restore(latest_checkpoint_path)
    #     #status.assert_consumed()
    #
    #
    #     # Display value after checkpoint restore
    #     for layer in energy_model.trainable_variables:
    #         print(f'energy_model weights after checkpoint assignment, {layer}')
    #         break
    #
    #     #saved_model_path = 'Sim_states_reduced_domain_5/policies/policy'
    #     #checkpoint_path = 'Sim_states_reduced_domain_5/train/checkpoints/'
    #
    #     # policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    #     #    saved_model_path, load_specs_from_pbtxt=True)
    #     # policy.update_from_checkpoint(checkpoint_path)
    #     # print(f'policy {policy}, {type(policy)}')
    #     #energy_model = policy
    #
    #     # # Compare the variables in the policy to the variables in the checkpoint
    #     # for variable in energy_model.variables:
    #     #     checkpoint_variable = checkpoint.get_object(variable.name)
    #     #     if not tf.math.equal(variable, checkpoint_variable):
    #     #         print(f"Variable {variable.name} does not match the checkpoint")
    #
    #
    #     # # Specify path of saved model
    #     #saved_model_dir = "Sim_states_pretrained_3/policies/collect_policy" #"my_saved_model_test/policy"
    #     # # Specify path of checkpoint
    #     # checkpoint_path = "Sim_states_pretrained_3/train/checkpoints/ckpt-10000"
    #     # checkpoint_dir = 'Sim_states_pretrained_3/train/checkpoints'
    #     #
    #     # # Load a saved model
    #     #model = tf.saved_model.load(saved_model_dir)
    #     # list(model.signatures.keys())
    #     #
    #     # # Display value before checkpoint restore
    #     # for var in model.model_variables[2:]:
    #     #     print(f'Before assignment: {var}')
    #     #     break
    #     # for layer in energy_model.trainable_variables:
    #     #     print(f'energy_model weights before checkpoint assignment, {layer}')
    #     #     break
    #     #
    #     # # restore the weights from a checkpoint
    #     # checkpoint = tf.train.Checkpoint(model=energy_model)  # TODO: This doesn't load any checkpoint weights
    #     # #status = checkpoint.restore(checkpoint_path)
    #     # status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #     #
    #     # for layer in energy_model.trainable_variables:
    #     #     print(f'energy_model weights after checkpoint assignment, {layer}')
    #     #     break
    #     #
    #     # # load the variables from the saved model into the new model
    #     # for var, saved_var in zip(energy_model.trainable_variables, model.model_variables[2:]):
    #     #     # This manually compares layers and sets the weights
    #     #     #print(f'model_variables after assignment, {saved_var}')
    #     #     var.assign(saved_var)
    #     #
    #     # for layer in energy_model.trainable_variables:
    #     #     print(f'CHECK: energy_model weights after assignment, {layer}')
    #     #     break
    #     #
    #     # #print(f'checkpoint summary, {tf.train.load_checkpoint(checkpoint_dir).get_variable_to_shape_map()}')
    #     # # print(f'model.model_variables summary, {model.model_variables.summary()}')
    #     #
    #     # #loaded = tf.saved_model.load(path)
    #     # #print("EBMMLP has {} trainable variables: {}, ...".format(
    #     # #    len(loaded.trainable_variables),
    #     # #    ", ".join([v.name for v in loaded.trainable_variables[:5]])))
    #     #
    #     # energy_model = model
    #
    #     # print(f'model: {dir(model)}')
    #     # #print(f'model.model_variables: {model.model_variables}')
    #     # print(f'model.cloning_network: {model.cloning_network}')
    #     # print(f'model.signatures: {model.signatures}')
    #     # print(f'model._export_to_saved_model_graph: {model._export_to_saved_model_graph}')
    #
    #     #energy_model.summary()

    #energy_model = model.cloning_network

    # # Loop through all layers in the energy model
    # for layer in energy_model.layers:
    #     print(f'layer: {layer}')
    #     # Check if the layer has any trainable weights
    #     if len(layer.trainable_weights) > 0:
    #         # Get the weights for this layer from the SavedModel
    #         layer_weights = [model.variables[i] for i in range(len(layer.weights))]
    #         print(f'layer_weights: {layer_weights}')
    #         # Set the weights for this layer in the energy model
    #         layer.set_weights(layer_weights)

    # Load the weights into the Keras model
    #energy_model.set_weights(model.model_variables)


    # Load the SavedModel
    # model = tf.saved_model.load(saved_model_dir)
    # loaded_variables = model.model_variables
    # print(f'loaded_variables: {loaded_variables}')
    # # Load the weights into the Keras model
    # #model.load_weights('Sim_states_pretrained_2/policies/policy/variables')
    # model.set_weights('Sim_states_pretrained_2/policies/policy/variables')

    """Create MLP with image features."""
    #energy_model = PixelEBM(obs_spec=obs_tensor_spec, action_spec=action_tensor_spec)
    # PixelEBM(obs_spec=obs_tensor_spec, action_spec=action_tensor_spec)

    # print(f'energy_model: {type(energy_model)}')
    # print(f'energy_model dir: {dir(energy_model)}')

    # if saved_model_dir is not None:

    return energy_model
