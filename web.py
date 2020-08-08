import sys
import glob
import numpy as np
import tensorflow as tf
from warnings import simplefilter
from keras import backend as K
from keras.models import model_from_json
from flask import Flask, jsonify
from flask_cors import CORS

from MinerEnv import MinerEnv
from predict.deep_q_network import DQNPredict


simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=config))

json_path = max(glob.glob('TrainedModels/*.json'))
model = DQNPredict(json_path)
print("Loaded model %s from disk" % json_path)

status_map = {0: "PLAYING", 1: "ELIMINATED WENT OUT MAP", 2: "ELIMINATED OUT OF ENERGY",
              3: "ELIMINATED INVALID ACTION", 4: "STOP EMPTY GOLD", 5: "STOP END STEP"}

# Initialize environment
minerEnv = MinerEnv(None, None)
minerEnv.start()
minerEnv.send_map_info("map1,0,0,50,100")
minerEnv.reset()
s = minerEnv.get_state()


@app.route('/next', methods=['GET'])
def get_next():
    global s
    if not minerEnv.check_terminate():
        action = np.argmax(model.predict(s))
        minerEnv.step(str(action))
        s = minerEnv.get_state()
        return jsonify({
            'state': minerEnv.get_readable_state(),
            'status': status_map[minerEnv.state.status],
            'action': int(action),
        })
    return jsonify({
        'status': status_map[minerEnv.state.status],
    })


@app.route('/reset', methods=['POST'])
def reset():
    global s
    minerEnv.send_map_info("map1,0,0,50,100")
    minerEnv.reset()
    s = minerEnv.get_state()
    return jsonify({
        'state': minerEnv.get_readable_state(),
        'status': status_map[minerEnv.state.status],
    })
