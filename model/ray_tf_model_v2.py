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
        # Configs
        custom_model_config = model_config['custom_model_config']
        max_step = custom_model_config['max_step']
        embedding_size = custom_model_config['embedding_size']
        conv1_filters = custom_model_config['conv1_filters']

        # View input
        input_view = tf.keras.layers.Input(shape=(21, 9))
        object_view = tf.keras.layers.Reshape(
            target_shape=(21 * 9,))(input_view)
        embedding_object_view = tf.keras.layers.Embedding(
            input_dim=38,
            output_dim=embedding_size,
        )(object_view)
        embedding_object_view = tf.keras.layers.Reshape(
            target_shape=(21, 9, embedding_size))(embedding_object_view)

        # Players input
        input_players = tf.keras.layers.Input(shape=(max_step, 4))
        input_players_int = tf.cast(input_players, tf.int32)
        input_single_player_list = tf.unstack(input_players_int, axis=-1)
        per_player_pos_onehot_list = []
        for input_single_player in input_single_player_list:
            per_player_pos_onehot = tf.one_hot(input_single_player, depth=21 * 9)
            per_player_pos_onehot = tf.reshape(tf.transpose(per_player_pos_onehot, [0, 2, 1]), [-1, 21, 9, max_step])
            per_player_pos_onehot_list.append(per_player_pos_onehot)

        # Combine view & players inputs
        per_player_view_list = []
        for per_player_pos_onehot in per_player_pos_onehot_list:
            per_player_view_list.append(tf.expand_dims(
                tf.concat([embedding_object_view, per_player_pos_onehot], axis=-1),
                axis=1))
        player_view = tf.concat(per_player_view_list, axis=1)

        cnn = tf.keras.Sequential(name='conv1')
        for filters, kernel_size, strides in conv1_filters:
            cnn.add(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation='elu'))
        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(256, activation='elu'))

        cnn_out = tf.keras.layers.TimeDistributed(cnn)(player_view)
        flatten = tf.keras.layers.Flatten()(cnn_out)
        model1_output = tf.keras.layers.Dense(384, activation='elu')(flatten)
        model1 = tf.keras.Model(inputs=[input_view, input_players], outputs=model1_output, name='view')

        input_energy = tf.keras.layers.Input(shape=(4,))
        model2_output = tf.keras.layers.Dense(128, activation='elu')(input_energy)
        model2 = tf.keras.Model(inputs=input_energy, outputs=model2_output, name='energy')

        concat = tf.concat([model1.output, model2.output], axis=-1)

        output = tf.keras.layers.Dense(num_outputs, activation='elu')(concat)

        self.base_model = tf.keras.Model(
            inputs=[input_view, input_players, input_energy], outputs=output)
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state
