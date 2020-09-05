import unittest
from unittest.mock import patch, mock_open
import json

from GAME_SOCKET_DUMMY import GameInfo, GameSocket, GoldInfo, ObstacleInfo


class TestGameInfo(unittest.TestCase):
    def test_init(self):
        game_info = GameInfo(4, 100)

        self.assertEqual(4, game_info.numberOfPlayers)
        self.assertEqual(100, game_info.steps)
    
    def test_load_map(self):
        game_info = GameInfo(4, 100)
        game_info.load_map([[450, 0, 0],
                            [-1, -2, -3]])
        
        self.assertEqual(3, game_info.width)
        self.assertEqual(2, game_info.height)
        self.assertEqual(1, len(game_info.golds))
        self.assertEqual(0, game_info.golds[0].posx)
        self.assertEqual(0, game_info.golds[0].posy)
        self.assertEqual(450, game_info.golds[0].amount)
        self.assertEqual(5, len(game_info.obstacles))


map_str = '''[[450, 0],
[-2, -1]]'''


class TestGameSocket(unittest.TestCase):
    def setUp(self):
        self.mock_open_patcher = patch(
            'builtins.open', mock_open(read_data=map_str), create=True)
        self.mock_open = self.mock_open_patcher.start()
        self.maxDiff = None
    
    def test_reset(self):
        socket = GameSocket(None, None)
        socket.reset(['map1', '0', '0', '10', '3', '20'])

        self.assertEqual(0, socket.userMatch.posx)
        self.assertEqual(0, socket.userMatch.posy)
        self.assertEqual(10, socket.userMatch.energy)
        self.assertEqual(3, socket.userMatch.gameinfo.numberOfPlayers)
        self.assertEqual(20, socket.userMatch.gameinfo.steps)
        self.assertEqual(2, socket.userMatch.gameinfo.width)
        self.assertEqual(2, socket.userMatch.gameinfo.height)
        self.assertEqual(1, len(socket.userMatch.gameinfo.golds))
        self.assertEqual(3, len(socket.userMatch.gameinfo.obstacles))
        self.assertEqual(3, len(socket.users))

    @patch.object(GameSocket, 'reset')
    def test_send_reset(self, mock_reset):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,3,20')

        mock_reset.assert_called_with(['map1', '0', '0', '10', '3', '20'])
    
    @patch.object(GameSocket, 'reset')
    def test_send_update(self, mock_reset):
        socket = GameSocket(None, None)
        socket.send('1')

        mock_reset.assert_not_called()
    
    def test_send_reset_receive(self):
        socket = GameSocket(None, None)
        socket.send('map1,7,9,10,4,20')

        data = json.loads(socket.receive())
        self.assertDictContainsSubset({
            'posx': 7,
            'posy': 9,
            'energy': 10,
        }, data)
        self.assertDictContainsSubset({
            'numberOfPlayers': 4,
            'width': 2,
            'height': 2,
            'steps': 20,
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450,
                }
            ],
        }, data['gameinfo'])
        self.assertListEqual([
            {
                'type': 0,
                'posx': 1,
                'posy': 0,
                'value': -1
            },
            {
                'type': 2,
                'posx': 0,
                'posy': 1,
                'value': -10
            },
            {
                'type': 1,
                'posx': 1,
                'posy': 1,
                'value': 0
            }
        ], data['gameinfo']['obstacles'])

    def test_go_left(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': -1,
                    'posy': 0,
                    'score': 0,
                    'energy': 10,
                    'status': 1,
                    'lastAction': 6,
                    'freeCount': 0,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450
                }
            ],
            'changedObstacles': []
        }, data)

    def test_go_right(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 1,
            '2': 1,
            '3': 1,
            '4': 1,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': 1,
                    'posy': 0,
                    'score': 0,
                    'energy': 9,
                    'status': 0,
                    'lastAction': 1,
                    'freeCount': 0,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450
                }
            ],
            'changedObstacles': []
        }, data)

    def test_go_up(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 2,
            '2': 2,
            '3': 2,
            '4': 2,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': 0,
                    'posy': -1,
                    'score': 0,
                    'energy': 10,
                    'status': 1,
                    'lastAction': 6,
                    'freeCount': 0,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450
                }
            ],
            'changedObstacles': []
        }, data)

    def test_go_down(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 3,
            '2': 3,
            '3': 3,
            '4': 3,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': 0,
                    'posy': 1,
                    'score': 0,
                    'energy': 0,
                    'status': 2,
                    'lastAction': 6,
                    'freeCount': 0,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450
                }
            ],
            'changedObstacles': [
                {
                    'type': 0,
                    'posx': 0,
                    'posy': 1,
                    'value': -1,
                }
            ]
        }, data)

    def test_free(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 4,
            '2': 4,
            '3': 4,
            '4': 4,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': 0,
                    'posy': 0,
                    'score': 0,
                    'energy': 10,
                    'status': 0,
                    'lastAction': 4,
                    'freeCount': 1,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 450
                }
            ],
            'changedObstacles': []
        }, data)
    
    def test_craft(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 5,
            '2': 5,
            '3': 5,
            '4': 5,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': playerId,
                    'posx': 0,
                    'posy': 0,
                    'score': 50,
                    'energy': 5,
                    'status': 0,
                    'lastAction': 5,
                    'freeCount': 0,
                } for playerId in [1, 2, 3, 4]
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 250
                }
            ],
            'changedObstacles': []
        }, data)
    
    def test_mixed_actions(self):
        socket = GameSocket(None, None)
        socket.send('map1,0,0,10,4,20')
        socket.send(json.dumps({
            '1': 0,
            '2': 3,
            '3': 4,
            '4': 5,
        }))

        data = json.loads(socket.receive())
        self.assertDictEqual({
            'players': [
                {
                    'playerId': 1,
                    'posx': -1,
                    'posy': 0,
                    'score': 0,
                    'energy': 10,
                    'status': 1,
                    'lastAction': 6,
                    'freeCount': 0,
                },
                {
                    'playerId': 2,
                    'posx': 0,
                    'posy': 1,
                    'score': 0,
                    'energy': 0,
                    'status': 2,
                    'lastAction': 6,
                    'freeCount': 0,
                },
                {
                    'playerId': 3,
                    'posx': 0,
                    'posy': 0,
                    'score': 0,
                    'energy': 10,
                    'status': 0,
                    'lastAction': 4,
                    'freeCount': 1,
                },
                {
                    'playerId': 4,
                    'posx': 0,
                    'posy': 0,
                    'score': 50,
                    'energy': 5,
                    'status': 0,
                    'lastAction': 5,
                    'freeCount': 0,
                },
            ],
            'golds': [
                {
                    'posx': 0,
                    'posy': 0,
                    'amount': 400
                }
            ],
            'changedObstacles': [
                {
                    'type': 0,
                    'posx': 0,
                    'posy': 1,
                    'value': -1,
                }
            ]
        }, data)


    def tearDown(self):
        self.mock_open_patcher.stop()
