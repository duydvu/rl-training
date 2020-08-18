import numpy as np


def preprocess_state(state):
    view, players = state
    model_view = np.zeros((view.shape[0], view.shape[1], 5), dtype=float)
    model_view[:, :, 0] = view
    model_view[model_view > 0] = model_view[model_view > 0] / 1000
    for player in players:
        if player['id'] == 1:
            energy = player['energy'] / 50
        x = player['x']
        y = player['y']
        if x < view.shape[0] and y < view.shape[1]:
            model_view[x, y, player['id']] = 1
    return [model_view, np.array([energy])]