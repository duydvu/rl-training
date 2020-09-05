import os
import numpy as np
import importlib
from warnings import simplefilter
from flask import Flask, jsonify, request
from flask_cors import CORS
from random import randint

from src.game.miner_env import MinerEnv
from src.policies.bot1 import Bot1
from src.policies.bot2 import Bot2

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
state = miner_env.state
s = None
playerId = 1
step = 0
bot1 = Bot1(2)
bot2 = Bot1(3)
bot3 = Bot2(4)


@app.route('/next', methods=['GET'])
def get_next():
    global s, step
    if state.players[playerId]['status'] == 0:
        action = predictor.compute_action(s)
        miner_env.step({
            '1': action,
            '2': bot1.compute_action(s),
            '3': bot2.compute_action(s),
            '4': bot3.compute_action(s),
        })
        s = miner_env.get_state()
        step += 1
        return jsonify({
            'state': miner_env.get_readable_state(),
            'status': status_map[state.players[playerId]['status']],
            'action': int(action),
            'step': step,
        })
    return jsonify({
        'status': status_map[state.players[playerId]['status']],
    })


@app.route('/reset', methods=['POST'])
def reset():
    global s, step
    body = request.get_json()
    map_id = int(body.get('map_id', 1))
    x = int(body.get('init_x', 0))
    y = int(body.get('init_y', 0))
    miner_env.send_map_info(map_id, x, y, number_of_players=4)
    miner_env.reset()
    s = miner_env.get_state()
    step = 0
    predictor.reset()
    bot1.reset()
    bot2.reset()
    bot3.reset()
    return jsonify({
        'state': miner_env.get_readable_state(),
        'status': status_map[state.players[playerId]['status']],
    })
