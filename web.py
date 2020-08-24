import os
import numpy as np
import importlib
from warnings import simplefilter
from flask import Flask, jsonify, request
from flask_cors import CORS
from random import randint

from MinerEnv import MinerEnv

simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)

predictor = getattr(importlib.import_module(os.getenv('PREDICTOR_MODULE')),
    os.getenv('PREDICTOR_CLASS'))()

status_map = {0: "PLAYING", 1: "ELIMINATED WENT OUT MAP", 2: "ELIMINATED OUT OF ENERGY",
              3: "ELIMINATED INVALID ACTION", 4: "STOP EMPTY GOLD", 5: "STOP END STEP"}

# Initialize environment
miner_env = MinerEnv(None, None)
miner_env.start()
s = None
playerId = 1

@app.route('/next', methods=['GET'])
def get_next():
    global s
    if miner_env.state.status[playerId] == 0:
        action = predictor.compute_action(s)
        miner_env.step({
            '1': action,
            '2': randint(0, 5),
            '3': randint(0, 5),
            '4': randint(0, 5),
        })
        s = miner_env.get_state()
        return jsonify({
            'state': miner_env.get_readable_state(),
            'status': status_map[miner_env.state.status[playerId]],
            'action': int(action),
        })
    return jsonify({
        'status': status_map[miner_env.state.status[playerId]],
    })


@app.route('/reset', methods=['POST'])
def reset():
    global s
    body = request.get_json()
    map_id = int(body.get('map_id', 1))
    x = int(body.get('init_x', 0))
    y = int(body.get('init_y', 0))
    miner_env.send_map_info(map_id, x, y)
    miner_env.reset()
    s = miner_env.get_state()
    return jsonify({
        'state': miner_env.get_readable_state(),
        'status': status_map[miner_env.state.status[playerId]],
    })
