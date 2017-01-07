#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Design and handle multilayer graphs."""

import networkx as nx
import numpy as np
import pandas as pd

def say_hello(name):
	print('Hello Mister {}'.format(name))

##################################################################
# Graph design

def add_list_of_words(G,list_of_words):
	""" add a list of words to the graph G and connect them together (fully connected subgraph).
		If some of the words already exist and are linked, increase the edge weight by one.
		Return the updated graph
	"""
	import itertools
	#G = Graph.copy()
	wordset = set(list_of_words)
	if len(wordset)>0:
		couples = itertools.combinations(wordset, 2)
		#G.add_edges_from(edge_list)
		for edge in couples:
			if G.has_edge(edge[0],edge[1]):
				# we added this one before, just increase the weight by one
				G[edge[0]][edge[1]]['weight'] += 1
			else:
				# new edge. add with weight=1
				G.add_edge(edge[0], edge[1], weight=1)
	return G

def add_string_of_words(G,list_of_words):
	""" add a string of words to the graph G and connect them in following order.
		If some of the words already exist and are linked, 
		increase the edge weight by one and add a new edge id.
		Return the updated graph
	"""
	#G = Graph.copy()
	for word in list_of_words:
		for edge in couples:
			if G.has_edge(edge[0],edge[1]):
				# we added this one before, just increase the weight by one
				G[edge[0]][edge[1]]['weight'] += 1
			else:
				# new edge. add with weight=1
				G.add_edge(edge[0], edge[1], weight=1)
	return G

#######################################################################
# Handle attributes
#######################################################################
def compute_degrees(G,kind='weighted',weight='weight'):
	""" Compute the degree and store it as node properties.
		kind (string, default 'weighted'): specify 'weighted', 'unweighted' or 'both'
		weight (string, default 'weight'): specify the variable name where the weightsd are stored
	"""
	if kind == 'weighted':
		degreeDic = G.degree(weight=weight)
		nx.set_node_attributes(G,'degree_w',degreeDic)
	elif kind == 'unweighted':
		degreeDic = G.degree()
		nx.set_node_attributes(G,'degree',degreeDic)
	elif kind == 'both':
		degreeDic = G.degree(weight=weight)
		nx.set_node_attributes(G,'degree_w',degreeDic)
		degreeDic = G.degree()
		nx.set_node_attributes(G,'degree',degreeDic)
	else:
		raise ValueError("kind must be 'weighted', 'unweighted' or 'both'")

def normalize_weights(G,weight=None,weight_n='weight_n'):
	""" Compute the degree of all nodes and normalize the weights of a graph
		store the values in the property 'weight_n'
		G : graph
		weight : string or None (default=None), the variable name where the weights are stored.
		weight_n: string (default'weight_n'), the name under which the normalized weights are stored
	"""

	degreeDic = G.degree(weight=weight)
	nx.set_node_attributes(G,'degree',degreeDic)
	for node1,node2,data in G.edges(data=True):
		d1 = np.sqrt(G.node[node1]['degree'])
		d2 = np.sqrt(G.node[node2]['degree'])
		if weight==None:
			G[node1][node2][weight_n] = 1.0/d1/d2
		else:
			v_weight = G[node1][node2][weight]
			G[node1][node2][weight_n] = v_weight/d1/d2
	# optional: the normalized degree
	#degreeDic = G.degree(weight=weight_n)
	#nx.set_node_attributes(G,'degree_n',degreeDic)
	return G

def rescale_weights(G,weight='weight',weight_rescale='weight_rescale'):
	""" Rescale the edge weights (divide by the max weight),
		and store the value in the variable 'weight_norm'. 
		weight, string (default='weight'): specify the name of the weight variable.
		weight_rescale, string (default='weight_rescale'): name of the variable to store the rescaled weight in.
	"""
	n1,n2,weights = zip(*G.edges(data=weight))
	edges_id=list(zip(n1,n2))
	print('mean weight: '+str(np.mean(weights))+', max weight: '+str(np.max(weights)))
	weights_n=weights/np.max(weights)
	weights_n_dic=dict(zip(edges_id,weights_n)) 
	nx.set_edge_attributes(G,weight_rescale,weights_n_dic)
	print('created variable {}'.format(weight_rescale))
	return G

def remove_weak_links(G,threshold,weight='weight_n'):
	""" Remove the weakest edges (weight smaller than threshold) of the most connected nodes of G
		use the weights stored in variable name weight (default='weight_n')
	"""
	print('Initial size: {}'.format(G.size()))
	for u,v,a in G.edges(data=True):
		if a[weight]<threshold:
			G.remove_edge(u,v)
	print('Final size: {}'.format(G.size()))
	return G

def shrink_graph(G,max_nb_of_edges,start=0,step=0.001):
	""" reduce the number of edges of graph G
		to less than max_nb_of_edges.
		the weakest links are removed (from weight_n)
		start (default=1): initial threshold
		step (default=0.001): specify the threshold step
		return shrinked graph
	"""
	H = G.copy()
	threshold = start
	while H.size() > max_nb_of_edges:
		print(threshold)
		H = remove_weak_links(H,threshold)
		threshold +=step
	return H

def merge_strongly_connected_nodes(G,ratio=0.5,iterations=2):
	""" find the strongest connections in the graph and
		merge the corresponding nodes. For the nodes to be merged,
		the weight of their connection must be larger than
		ratio * (sum(weight of node 1)+sum(weight of node 2))/2
		G : graph
		ratio (int, default=0.5): the threshold value
		iterations (int, default=2): do several passes of merge, 
									the number is given by iterations. 
	"""
	G = normalize_weights(G,weight='weight')
	for i in range(iterations):
		# extract the edge weights from the graph and sort them in ascending order
		edges = list(G.edges_iter(data='weight_n', default=1))
		df = pd.DataFrame(edges,columns=['source','target','weight_n'])
		df = df.sort_values('weight_n',ascending=False)
		#df_i = df.iloc[0:1000,:]
		# merge the nodes
		G = merge_nodes_from_df(G,df,ratio)
		#print('== next ==')   

def merge_nodes_from_df(G,df,ratio=0.5):
	""" merge nodes in the df if the connection is above a threshold
		related to the value of 'ratio'
		df: pandas dataframe
		ratio (int, default=0.5)
	"""
	# Iterate over the rows of the dataframe
	for row in df.itertuples():
		source = row.source
		target = row.target
		weight = row.weight_n
		if source in G and target in G:
			neig_s = G.neighbors(source)
			neig_t = G.neighbors(target)
			sum_s = sum([G[source][n]['weight_n'] for n in neig_s])
			sum_t = sum([G[target][n]['weight_n'] for n in neig_t])
			nb_neig_s = G.degree(source)
			nb_neig_t = G.degree(target)
			if 2*weight/(sum_s+sum_t)>ratio:#/(min(nb_neig_s,nb_neig_t)):
				print(source,target,' merged.')#,weight,sum_s,sum_t,nb_neig_s,nb_neig_t)
				G = merge_nodes(G,source,target,data=True)
			else:
				pass
				#print(source,target,'=not strong enough=',weight,sum_s,sum_t,nb_neig_s,nb_neig_t)
		else:
			pass
			#print(source,target,'=node merged=')
	return G

def merge_nodes(G, node1, node2, data=False):
	""" merge two nodes node1 and node2 in the graph into one node 
		with id string node1+'_'+node2
		if data=None, any data is ignored
		if data='node1' or 'node2' the new node inherit the data of the given node and
		the degree of all nodes is recomputed after merging
	"""
	H = G.copy()
	# create the new node
	node_id = node1+'_'+node2
	if data == False:
		H.add_node(node_id)
	elif data == True:
		degree1 = len(G[node1])
		degree2 = len(G[node2])
		if degree1 > degree2:
			H.add_node(node_id, H.node[node1])
		else:
			H.add_node(node_id, H.node[node2])
	else:
		raise ValueError("data only accept True or False")
	# connect it to the rest
	for n, n_data in H[node1].items():
		if not (n == node2 or n == node1):
			#props = H[node1][n]
			H.add_edge(node_id, n, n_data)
	for n, n_data in H[node2].items():
		if not (n == node1 or n == node2):
			#props = H[node2][n]
			H.add_edge(node_id, n, n_data)
	# remove the initial nodes and edges
	H.remove_node(node1)
	H.remove_node(node2)
	# compute new nodes properties
	# TODO: recompute only for the neighbors of the merged nodes
	H = normalize_weights(H,weight='weight')
	return H

################################################################
# Ouput the graph

def save_json(G,filename):
	""" Write the graph to a json file."""
	from networkx.readwrite import json_graph
	datag = json_graph.node_link_data(G)
	import json
	s = json.dumps(datag)
	datag['links'] = [
		{
			'source': datag['nodes'][link['source']]['id'],
			'target': datag['nodes'][link['target']]['id'],
			'weight': link['weight'],
			'weight_n': link['weight_n']
		}
		for link in datag['links']]
	s = json.dumps(datag)
	with open(filename, "w") as f:
		f.write(s)
