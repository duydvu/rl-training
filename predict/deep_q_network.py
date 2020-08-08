from keras.models import model_from_json
import numpy as np

from model.deep_q_network import DQN


class DQNPredict:
    def __init__(self, json_path):
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights('%s.h5' % json_path[:-5])

    @staticmethod
    def transform_state(state):
        return DQN.transform_state(state)

    def predict(self, state):
        model_state = self.transform_state(state)
        return np.argmax(self.model.predict(
            [[model_state[0]], [model_state[1]]]))
