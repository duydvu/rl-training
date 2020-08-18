import json
import pandas as pd
import numpy as np
import tensorflow as tf
from random import randrange

from model.policy_gradient_network import PGNetwork
from MinerEnv import MinerEnv
from utils import get_now_str
from recorder import Recorder
from transformer import transform_state


config = json.load(open('config/v1.json'))

tf.reset_default_graph()

view_state_size = [5, 3, 5]
player_state_size = 1
action_size = 6
model = PGNetwork(view_state_size, player_state_size,
                  action_size, filters=[(64, 3, 2)], lstm_state_size=256, hidden_layer_size=64, learning_rate=5e-3)

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
session = tf.Session(config=session_config)
session.run(tf.global_variables_initializer())

# Initialize environment
minerEnv = MinerEnv(host=None, port=None)
minerEnv.start()

gamma = 0.9

def new_random_episode():
    # Choosing a map in the list
    map_id = np.random.randint(0, 1)
    pos_x = np.random.randint(0, 1)
    pos_y = np.random.randint(0, 1)
    # pos_x = np.random.randint(config['MAP_MAX_X'])
    # pos_y = np.random.randint(config['MAP_MAX_Y'])
    minerEnv.send_map_info(map_id, pos_x, pos_y, max_steps=100)
    minerEnv.reset()


def discount_and_normalize_rewards(rewards, episode_len):
    discounted_rewards = np.zeros_like(rewards)
    cumulative = 0.0
    for i in reversed(range(episode_len)):
        cumulative = cumulative * gamma + rewards[i]
        discounted_rewards[i] = cumulative

    # mean = np.mean(discounted_rewards[:episode_len])
    # std = np.std(discounted_rewards[:episode_len]) + 1e-12
    # discounted_rewards[:episode_len] = (discounted_rewards[:episode_len] - mean) / (std)

    return discounted_rewards


def make_batch(batch_size):
    view_states_batch = []
    player_states_batch = []
    actions_batch = []
    rewards_batch = [] 
    discounted_rewards_batch = []
    episode_lens_batch = []
    first_prob = None
    for _ in range(batch_size):
        new_random_episode()
        maxStep = minerEnv.state.mapInfo.maxStep
        view_states_episode = np.zeros((maxStep, *view_state_size))
        player_states_episode = np.zeros((maxStep, player_state_size))
        actions_episode = np.zeros((maxStep, action_size))
        rewards_episode = np.zeros((maxStep))
        action = None
        lstm_state = np.zeros((2, 1, 256), dtype=float)
        for step in range(maxStep):
            state = minerEnv.get_state()
            # Run State Through Policy & Calculate Action
            view_state, player_state = transform_state(state)
            action_prob_distribution, lstm_state = session.run([model.action_distribution, model.lstm_state],
                                                                feed_dict={
                                                                    model.view_inputs: view_state.reshape(1, 1, *view_state_size),
                                                                    model.player_inputs: player_state.reshape(1, 1, player_state_size),
                                                                    model.length: [1],
                                                                    model.state_placeholder: lstm_state})
            action_prob_distribution = action_prob_distribution.ravel()
            if first_prob is None:
                first_prob = action_prob_distribution
            action = np.random.choice(action_size,
                                      p=action_prob_distribution)
            action_onehot = np.zeros_like(action_prob_distribution)
            action_onehot[action] = 1

            # Perform action
            minerEnv.step(str(action))
            reward = minerEnv.get_reward()
            done = minerEnv.check_terminate()

            # Store results
            view_states_episode[step] = view_state
            player_states_episode[step] = player_state
            actions_episode[step] = action_onehot
            rewards_episode[step] = reward

            if done:
                episode_len = step + 1
                view_states_batch.append(view_states_episode)
                player_states_batch.append(player_states_episode)
                actions_batch.append(actions_episode)
                rewards_batch.append(rewards_episode)
                discounted_rewards_batch.append(discount_and_normalize_rewards(rewards_episode, episode_len))
                episode_lens_batch.append(episode_len)
                break

    return (
        np.stack(view_states_batch),
        np.stack(player_states_batch),
        np.stack(actions_batch),
        np.stack(rewards_batch),
        np.stack(discounted_rewards_batch),
        np.stack(episode_lens_batch),
        first_prob
    )


batch_size = 256
saver = tf.train.Saver()

max_reward = 0
batch_mean_rewards = []

if __name__ == '__main__':
    for epoch in range(0, config['N_EPISODE']):
        view_states_mb, player_states_mb, actions_mb, rewards_mb, discounted_rewards_batch, episode_lens_mb, first_prob = make_batch(
            batch_size)

        loss, _ = session.run([model.loss, model.train_op], feed_dict={
            model.view_inputs: view_states_mb,
            model.player_inputs: player_states_mb,
            model.actions: actions_mb,
            model.discounted_episode_rewards: discounted_rewards_batch,
            model.length: episode_lens_mb,
            model.state_placeholder: np.zeros((2, batch_size, 256))})

        total_rewards_of_mb = np.sum(rewards_mb, axis=1)
        mean_reward_of_mb = np.mean(total_rewards_of_mb)
        max_reward_of_mb = np.max(total_rewards_of_mb)
        min_reward_of_mb = np.min(total_rewards_of_mb)
        max_reward = max(max_reward, max_reward_of_mb)

        print("==========================================")
        print("Epoch: ", epoch)
        print("-----------")
        print("Mean reward of batch: {}".format(mean_reward_of_mb))
        print("Max reward of batch: {}".format(max_reward_of_mb))
        print("Min reward of batch: {}".format(min_reward_of_mb))
        print("Max reward recorded: {}".format(max_reward))
        print("Loss: {}".format(loss))
        print(first_prob)

        # Iteration to save the network architecture and weights
        if epoch % config['SAVE_NETWORK'] == 0:
            saver.save(session, './TrainedModels/model', global_step=epoch)
