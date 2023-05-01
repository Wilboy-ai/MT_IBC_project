import tensorflow as tf
from google.protobuf.descriptor import FieldDescriptor

# # Path to the .tfrecord file
# tfrecord_path = "data/2d_oracle_particle_0.tfrecord"
# spec_file_path = f"{tfrecord_path}.spec"
#
# # # Open the file and create a TFRecordDataset
# dataset = tf.data.TFRecordDataset(tfrecord_path)
#
# # Iterate over the dataset and print each example
# for record in dataset:
#     example = tf.train.Example()
#     example.ParseFromString(record.numpy())
#     print(example)



# # # Load the spec file and parse the protobuf message
# spec = tf.io.gfile.GFile(spec_file_path, 'rb').read()
# message = tf.train.Example.FromString(spec)
#
# # Extract the feature names and types
# feature_names = []
# feature_types = {}
# for feature in message.features.feature:
#     feature_name = feature.name
#     feature_names.append(feature_name)
#
#     # Determine the feature type
#     if feature.HasField('bytes_list'):
#         feature_type = 'bytes_list'
#     elif feature.HasField('float_list'):
#         feature_type = 'float_list'
#     elif feature.HasField('int64_list'):
#         feature_type = 'int64_list'
#     else:
#         raise ValueError(f"Unknown feature type in spec: {feature_name}")
#
#     feature_types[feature_name] = feature_type


# from load_data import create_sequence_datasets
# batch_size = 512
#
# # Replace 'path_to_your_tfrecord_files_directory' with the path to the directory containing your TFRecord files
# # Replace 'batch_size' with the desired batch size
# train_dataset = create_sequence_datasets(tfrecord_path, batch_size)
#
# # Load a batch of data
# for batch in train_dataset.take(1):
#     print(batch)

# import tensorflow as tf
#
# # Load the .tfrecord.spec file
# # with open("data/2d_oracle_particle_0.tfrecord.spec", "r", encoding="windows-1252") as f:
# #     spec_text = f.read()
# #     print(spec_text)
#
# with open("data/2d_oracle_particle_0.tfrecord.spec", "rb") as f:
#     spec_bytes = f.read()
#     spec = tf.train.Example.FromString(spec_bytes)
#     feature_description = {}
#     for k, v in spec.features.feature.items():
#         feature_description[k] = tf.io.FixedLenFeature(v.int64_list.value, v.dtype)
#     print(feature_description)




# import tensorflow as tf
# from tf_agents.trajectories import time_step as ts
# from tf_agents.trajectories import policy_step as ps
# from tf_agents.utils import example_encoding_dataset
#
# tf.compat.v1.disable_eager_execution()
#
# # Define the specs for the data that will be written to the .tfrecord file
# data_spec = {
#     'observation': tf.constant([1.0, 2.0]),
#     'action': tf.constant(0),
#     'reward': tf.constant(1.0),
#     'next_observation': tf.constant([3.0, 4.0]),
#     'step_type': tf.constant(0, dtype=tf.int32)
# }
#
# # Create a TFRecordWriter to write the data to a .tfrecord file
# file_path = 'data.tfrecord'
# writer = tf.io.TFRecordWriter(file_path)
#
# # Create a TFRecordObserver to observe the data and write it to the .tfrecord file
# observer = example_encoding_dataset.TFRecordObserver(file_path, data_spec)
#
# # Define some data to write to the .tfrecord file
# data = {
#     'observation': tf.constant([1.0, 2.0]),
#     'action': tf.constant(0),
#     'reward': tf.constant(1.0),
#     'next_observation': tf.constant([3.0, 4.0]),
#     'step_type': tf.constant(0)
# }
#
# # Create a TimeStep to represent the initial state of the episode
# initial_time_step = ts.restart(data['observation'])
#
# # Reorder the elements of the initial_time_step to match the structure of the data_spec
# initial_time_step = ts.TimeStep(
#     step_type=initial_time_step.step_type,
#     reward=initial_time_step.reward,
#     discount=1.0,
#     observation=initial_time_step.observation
# )
#
# # Add an additional batch dimension to the initial time step
# initial_time_step = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), initial_time_step)
#
# # Write the initial time step to the .tfrecord file
# observer(initial_time_step)
#
# # Create a PolicyStep to represent the action taken by the policy
# policy_step = ps.PolicyStep(action=data['action'], state=())
#
# # Create a TimeStep to represent the next state of the episode
# next_time_step = ts.transition(data['next_observation'], data['reward'], data['step_type'])
#
# # Write the policy step and next time step to the .tfrecord file
# observer(policy_step, next_time_step)
#
# # Close the TFRecordWriter
# writer.close()
#
# # Write the data_spec to a .tfrecord.spec file
# spec_path = 'data.tfrecord.spec'
# example_encoding_dataset.dump_tf_record_specs(data_spec, spec_path)

import tensorflow as tf
from tf_agents.specs import tensor_spec

# Define a tensor spec
spec = tensor_spec.BoundedTensorSpec((3,), tf.float32, minimum=-1.0, maximum=1.0)

# Define a module subclass with the tensor spec
class MyModule(tf.Module):
    def __init__(self, spec):
        self.spec = spec

# Instantiate the module and save it
module = MyModule(spec)
tf.saved_model.save(module, 'my_module')
imported = tf.saved_model.load('my_module')
print(imported.spec)