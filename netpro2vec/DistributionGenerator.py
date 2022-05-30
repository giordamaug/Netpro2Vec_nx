# @Time    : 13/06/2022
# @Author  : Maurizio Giordano and Ichcha Manipur
# @Email   : maurizio.giordano@cnr.it, oichcha.manipur@gmail.com
# @File    : DistributionGenerator.py

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
from typing import *
import re
from . import utils

class DistributionGenerator:
	"""
	Generator class for Node distance distribution and Transition probability
	matrices
	"""
	def __init__(self, distrib_type, graphs,common_bin_list=True,verbose=False):
		self.verbose=verbose
		self.tqdm = tqdm if self.verbose else utils.nop
		self.distrib_type = distrib_type
		self.graph_list = graphs
		self.bin_list = None
		self.calculate_bin_list = common_bin_list
		self.distrib_list = []
		self.matcher = re.compile('^tm[0-9]+')
		self.__run_distib_comp()

	def get_distributions(self):
		return self.distrib_list

	def __get_bins(self):
		max_diam = max([nx.diameter(g) for g in self.graph_list])
		self.bin_list = np.append(np.arange(0, max_diam + 1), float('inf'))

	def __get_node_distance_distr(self, g):
		if self.bin_list is None:
			self.bin_list = np.append(np.arange(0, nx.diameter(g) + 1), float(
				'inf'))

		num_nodes = g.number_of_nodes()
		# Find node-wise distances and their histogram
		if nx.is_weighted(g):
			d = [[b for a,b in sorted(sp[1].items())] for sp in nx.shortest_path_length(g, weights='weight')]
		else:
			d = [[b for a,b in sorted(sp[1].items())] for sp in nx.shortest_path_length(g)]
		h_g = np.array([np.histogram(d[x], bins=self.bin_list)[0] for x in
						range(0, num_nodes)])  # s.transpose()
		distrib_mat = h_g / (num_nodes - 1)
		self.distrib_list.append(csr_matrix(distrib_mat))

	def __get_transition_matrix(self, g, walk=1):

		if g.is_directed():
			mode_g = "OUT"
		else:
			mode_g = "ALL"
		if walk == 1:
			# find 1 walk transition matrix
			if nx.is_weighted(g):
				adj_g = nx.adjacency_matrix(g,weight='weight').todense()
			else:
				adj_g = nx.adjacency_matrix(g).todense()
			adj_g = np.array(adj_g.data)
			dw = adj_g.sum(axis=0)
			dw[dw == 0] = 1
			distrib_mat = adj_g / dw
		else:
			# find Transition matrices > 1 walk
			if nx.is_weighted(g):
				d = [[b for a,b in sorted(sp[1].items())] for sp in nx.shortest_path_length(g, weights='weight')]
			else:
				d = [[b for a,b in sorted(sp[1].items())] for sp in nx.shortest_path_length(g)]
			# ego(g, order = walk, nodes = V(g), mindist = walk, mode = mode_g)
			num_nodes = g.number_of_nodes()
			walk_distances = np.zeros((num_nodes, num_nodes))

			# find vertices for the specified walk parameter by indexing the
			# distances matrix by using the ego indices (ego_out)
			for i in g.nodes():
				ego = nx.ego_graph(g,i,radius=walk, undirected=nx.is_undirected(g), distance=walk)
				for node in ego:
					walk_distances[i, node] = d[i][node]
			dw = walk_distances.sum(axis=0)
			dw[dw == 0] = 1
			# Normalize adjacency matrix to get the specified walk transition
			# matrix
			distrib_mat = walk_distances / dw
		distrib_mat = distrib_mat.T
		self.distrib_list.append(csr_matrix(distrib_mat))

	def __run_distib_comp(self):
		if self.distrib_type == "ndd":
			utils.vprint("Calculating Node Distance Distribution...", verbose=self.verbose)
			if self.calculate_bin_list:
				self.__get_bins()
			[self.__get_node_distance_distr(g) for g in self.tqdm(self.graph_list)]
		elif self.matcher.match(self.distrib_type):   # match any 'tm<int>'
			walk = int(self.distrib_type[2:])
			utils.vprint("Calculating Transition Matrices %s ..."%self.distrib_type, verbose=self.verbose)
			[self.__get_transition_matrix(g, walk=walk) for g in self.tqdm(self.graph_list)]
		else:
			Exception("Wrong distribution selection %r"%self.distrib_type)

def probability_aggregator_cutoff(probability_distrib_matrix, cut_off=0.01,
									agg_by=5, return_prob=True,
									remove_inf=False):
	if remove_inf:
		probability_distrib_matrix = np.delete(probability_distrib_matrix,
											   np.where(
												   probability_distrib_matrix[:,
												   -1] >= 1), axis=0)
	if agg_by > 0:
		# for pandas df
		# probability_distrib_matrix = pd.DataFrame(np.add.reduceat(
		# 	probability_distrib_matrix.values, np.arange(len(
		# 		probability_distrib_matrix.columns))[::agg_by], axis=1))
		probability_distrib_matrix = pd.DataFrame(np.add.reduceat(
			probability_distrib_matrix, np.arange(np.shape(
				probability_distrib_matrix)[1])[::agg_by], axis=1))
		probability_distrib_matrix = probability_distrib_matrix.to_numpy()
	if cut_off > 0:
		probability_distrib_matrix[probability_distrib_matrix <
								   cut_off] = 0
	if return_prob:
		return probability_distrib_matrix
