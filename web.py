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


simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=config))

model = max(glob.glob('TrainedModels/*.json'))
# load json and create model
with open(model, 'r') as json_file:
    loaded_model_json = json_file.read()
DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
DQNAgent.load_weights('%s.h5' % model[:-5])
print("Loaded model %s from disk" % model)

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
    if not minerEnv.check_terminate()[0]:
        action = np.argmax(DQNAgent.predict([[s[0]], [s[1]]]))
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
