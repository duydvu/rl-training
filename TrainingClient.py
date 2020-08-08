import logging
from model.deep_q_network import DQN
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process

import pandas as pd
import numpy as np
import json
from random import randrange

from utils import get_now_str
from recorder import Recorder

logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = json.load(open('config/v1.json'))

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(config['INPUT_NUM'], config['ACTION_NUM'], epsilon_decay=0.999)
memory = Memory(config['MEMORY_SIZE'])

recorder = Recorder(["Ep", "Step", "Reward",
                     "Total_reward", "Action", "Epsilon", "Done"])

# Initialize environment
minerEnv = MinerEnv(host=None, port=None)
minerEnv.start()

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.

for episode_i in range(0, config['N_EPISODE']):
    try:
        # Choosing a map in the list
        map_id = np.random.randint(1, 6)
        pos_x = np.random.randint(config['MAP_MAX_X'])
        pos_y = np.random.randint(config['MAP_MAX_Y'])
        minerEnv.send_map_info(map_id, pos_x, pos_y)

        # Getting the initial state
        minerEnv.reset()
        s = minerEnv.get_state()
        total_reward = 0
        maxStep = minerEnv.state.mapInfo.maxStep

        # Start an episde for training
        for step in range(0, maxStep):
            action = DQNAgent.act(s)
            minerEnv.step(str(action))
            s_next = minerEnv.get_state()
            reward = minerEnv.get_reward()
            terminate = minerEnv.check_terminate()

            # Add this transition to the memory batch
            memory.push(s, action, reward, terminate, s_next)

            # Sample batch memory to train network
            if (memory.length > config['INITIAL_REPLAY_SIZE']):
                #If there are INITIAL_REPLAY_SIZE experiences in the memory batch then start replaying
                batch = memory.sample(config['BATCH_SIZE'])
                DQNAgent.replay(batch, config['BATCH_SIZE'])
                train = True
            total_reward = total_reward + reward
            s = s_next

            # Saving data to file
            recorder.append([episode_i + 1, step + 1, reward,
                             total_reward, action, DQNAgent.epsilon, terminate])

            if terminate == True:
                break

        # Iteration to save the network architecture and weights
        if ((episode_i + 1) % config['SAVE_NETWORK'] == 0 and train == True):
            DQNAgent.target_train()
            DQNAgent.save_model("TrainedModels/DQNmodel_%s_ep%d" % (get_now_str(), episode_i + 1))

        
        # Print the training information after the episode
        logger.info('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f. Status code: %d',
                    episode_i + 1, step + 1, total_reward, DQNAgent.epsilon, minerEnv.state.status)
        
        # Decreasing the epsilon if the replay starts
        if train == True:
            DQNAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        break
