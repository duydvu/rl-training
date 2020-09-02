import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf


tf = try_import_tf()


class RayTFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        custom_model_config = model_config['custom_model_config']
        max_step = custom_model_config['max_step']
        embedding_size = custom_model_config['embedding_size']
        conv1_filters = custom_model_config['conv1_filters']
        conv2_filter = custom_model_config['conv2_filter']

        input_view = tf.keras.layers.Input(shape=(max_step, 21, 9, 2))
        input_players = tf.keras.layers.Input(shape=(max_step, 4))

        players_pos_onehot = tf.one_hot(tf.cast(input_players, tf.int32), depth=21 * 9)
        players_pos_onehot = tf.reshape(tf.transpose(players_pos_onehot, [0, 1, 3, 2]), [-1, max_step, 21, 9, 4])
        view = tf.concat([input_view, players_pos_onehot], axis=-1)

        object_view, *other_views = tf.unstack(view, axis=-1)
        object_view = tf.keras.layers.Reshape(
            target_shape=(max_step * 21 * 9,))(object_view)
        embedding_object_view = tf.keras.layers.Embedding(
            input_dim=20,
            output_dim=embedding_size,
        )(object_view)
        embedding_object_view = tf.keras.layers.Reshape(
            target_shape=(max_step, 21, 9, embedding_size))(embedding_object_view)
        other_views = tf.stack(other_views, axis=-1)
        conv_view = tf.concat([embedding_object_view, other_views], axis=-1)

        cnn = tf.keras.Sequential(name='conv1')
        for filters, kernel_size, strides in conv1_filters:
            cnn.add(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation='elu'))
        cnn.add(tf.keras.layers.Flatten())
        flatten = tf.keras.layers.TimeDistributed(cnn)(conv_view)
        model1_output = tf.keras.layers.Dense(256, activation='elu')(flatten)
        model1 = tf.keras.Model(inputs=[input_view, input_players], outputs=model1_output, name='view')

        input_energy = tf.keras.layers.Input(shape=(max_step, 4))
        model2_output = tf.keras.layers.Dense(256, activation='tanh')(input_energy)
        model2 = tf.keras.Model(inputs=input_energy, outputs=model2_output, name='energy')

        mul = tf.keras.layers.multiply([model1.output, model2.output])

        cnn2 = tf.keras.layers.Conv1D(
            filters=conv2_filter[0],
            kernel_size=conv2_filter[1],
            strides=conv2_filter[2],
            activation='elu',
            name='conv2')(mul)
        flatten2 = tf.keras.layers.Flatten()(cnn2)

        dense = tf.keras.layers.Dense(256, activation='relu')(flatten2)
        output = tf.keras.layers.Dense(num_outputs, activation='relu')(dense)

        self.base_model = tf.keras.Model(
            inputs=[input_view, input_players, input_energy], outputs=output)
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state
