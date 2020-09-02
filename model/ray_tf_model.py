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
        input_view = tf.keras.layers.Input(shape=(21, 9, 2))
        input_players = tf.keras.layers.Input(shape=(4))

        players_pos_onehot = tf.one_hot(tf.cast(input_players, tf.int32), depth=21 * 9)
        players_pos_onehot = tf.reshape(tf.transpose(players_pos_onehot, [0, 2, 1]), [-1, 21, 9, 4])
        view = tf.concat([input_view, players_pos_onehot], axis=-1)

        object_view, *other_views = tf.unstack(view, axis=-1)
        object_view = tf.keras.layers.Reshape(
            target_shape=(21 * 9,))(object_view)
        embedding_object_view = tf.keras.layers.Embedding(
            input_dim=9,
            output_dim=8,
        )(object_view)
        embedding_object_view = tf.keras.layers.Reshape(
            target_shape=(21, 9, 8))(embedding_object_view)

        other_views = tf.stack(other_views, axis=-1)
        conv_view = tf.concat([embedding_object_view, other_views], axis=-1)
        conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=4,
            strides=1,
            activation='elu')(conv_view)
        conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=2,
            activation='elu')(conv)
        conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=2,
            strides=2,
            activation='elu')(conv2)
        flatten = tf.keras.layers.Flatten()(conv3)
        y = tf.keras.layers.Dense(256, activation='elu')(flatten)
        model1 = tf.keras.Model(inputs=[input_view, input_players], outputs=y)

        input_energy = tf.keras.layers.Input(shape=(4,))
        model2_output = tf.keras.layers.Dense(
            256, activation='tanh')(input_energy)
        model2 = tf.keras.Model(
            inputs=input_energy, outputs=model2_output)

        mul = tf.keras.layers.multiply([model1.output, model2.output])

        output = tf.keras.layers.Dense(num_outputs, activation='relu')(mul)

        self.base_model = tf.keras.Model(
            inputs=[input_view, input_players, input_energy], outputs=output)
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state
