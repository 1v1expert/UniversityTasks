# -*- coding: utf-8 -*-
import pydot
import numpy.random as rand
import networkx as nx

from networkx.drawing.nx_pydot import write_dot

class PositionalGame(object):
	def __init__(self, players_count=0, tree_depth=0, tree=None):
		self.players_count = players_count
		self.tree_depth = tree_depth
		self.ways = []
 self.tree = tree
 if not tree:
 self.generate_random_tree()
 def generate_random_tree(self):
 self.tree = nx.balanced_tree(2, self.tree_depth - 1)
 for i in range(2 ** (self.tree_depth - 1) - 1, 2 ** self.tree_depth -
1):
 wins = rand.randint(-10, 10, self.players_count)
 self.tree.nodes[i]['wins'] = [tuple(wins)]
 def reverse_induction_step(self, parent_name, player=0):
 wins = []
 max_win = 0
 for name in self.tree[parent_name].keys():
 if 'wins' not in self.tree[name] and name > parent_name:
 self.reverse_induction_step(name, (player + 1) %
self.players_count)
 # ищем максимум в узле
 node_max_win = self.tree.nodes[name]['wins'][0][player]
 for win in self.tree.nodes[name]['wins']:
 if win[player] > node_max_win:
 node_max_win = win[player]
 # сравниваем максимум с другими узлами
 if not len(wins) or node_max_win == max_win:
 wins.extend(self.tree.nodes[name]['wins'])
 max_win = node_max_win
 elif node_max_win > max_win:
 wins = self.tree.nodes[name]['wins']
 max_win = node_max_win
 if len(wins):
 self.tree.nodes[parent_name]['wins'] = wins
 def solve(self):
 leafs = [name for name, node in self.tree.nodes.items() if 'wins' in
node]
 self.reverse_induction_step(0)
 for leaf in leafs:
 if self.tree.nodes[leaf]['wins'][0] in
self.tree.nodes[0]['wins']:
 self.ways.append(nx.shortest_path(self.tree, 0, leaf))
 def get_labels(self):