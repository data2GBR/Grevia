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

def add_string_of_words(G,list_of_words,text_id,text_data):
	""" add a string of words to the graph G and connect them in following order.
		If some of the words already exist and are linked, 
		increase the edge weight by one and add the details of the new path to the path
		dictionary attached to the edge.
		G: graph
		list_of_words (list of strings): list of text words
		text_id (num or string): text id
		text_data (dict): dictionary of text data
		Return the updated graph
	"""
	#G = Graph.copy()
	for idx,word in enumerate(list_of_words):
		# for each text in the dataframe, words are nodes of the graph,
		# connected if they follow each other
		#print(idx)
		path_props_dic = {}
		path_props_dic['text_id'] = text_id
		path_props_dic['text_data'] = text_data
		path_props_dic['word_positions'] = [idx]
		path_key = str(text_id)#+'_'+str(idx)
		if (idx+1)<len(list_of_words):
			word_id = word
			next_word_id = list_of_words[idx+1]
			# Adding the nodes
			if not G.has_node(word_id):
				G.add_node(word_id)
			if not G.has_node(next_word_id):
				G.add_node(next_word_id)
			# Adding an edge
			if G.has_edge(word_id,next_word_id):
				# we have already seen the edge before, 
				# just increase the weight by one
				G[word_id][next_word_id]['weight'] += 1
				# save the path details on the edge
				if path_key in G[word_id][next_word_id]['paths'].keys():
					G[word_id][next_word_id]['paths'][path_key]['word_positions'].append(idx)
				else:
					G[word_id][next_word_id]['paths'][path_key] = path_props_dic	
			else:
				# new edge. add with weight=1,
				# add the path details in the path dictionary
				dic_of_paths = {}
				dic_of_paths[path_key] = path_props_dic
				G.add_edge(word_id, next_word_id,
					{'weight':1,'paths':dic_of_paths})
	return G

#######################################################################
# Handle attributes
#######################################################################

def light_copy(G):
	""" return a new graph, copy of G but without the data on the edges and nodes
		except edges weights
	"""
	if nx.is_directed(G):
		H = nx.DiGraph()
	else:
		H = nx.Graph()
	#[H.add_node(node) for node in G.nodes()]
	H.add_nodes_from(G.nodes())
	for node1,node2,data in G.edges(data=True):
		data_light = {k:data[k] for k in data if not k=='paths'}
		#data_light = {'weight' : data['weight'],'weight_n' : data['weight_n'],'doc_freq' : data['doc_freq']}
		H.add_edge(node1,node2,data_light)
	return H

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

def compute_doc_freq(G):
	""" Compute the document frequency of an edge (nb of documents where it appears).
		save the value of each edge of the graph.
	"""
	for node1,node2,data in G.edges(data=True):
		doc_list=[ path_dic['text_id'] for path_dic in data['paths'].values()]
		nb_doc = len(set(doc_list))
		G[node1][node2]['doc_freq'] = nb_doc

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

def value_stats(G,item,value):
	""" return the mean, min and max of a value attached to the nodes or edges of graph G
		item (string): 'node' or 'edge'
		value (string): name of the value
	"""
	if item=='edge':
		n1,n2,data = zip(*G.edges(data=value))
	elif item=='node':
		n1,data_dic = zip(*G.nodes(data=True))
		data = [d[value] for d in data_dic]
	else:
		raise ValueError("only 'node' or 'edge' are accepted as item")
	mean_value = np.mean(data)
	min_value = np.min(data)
	max_value = np.max(data)
	print('statistics for value {}'.format(value))
	print('mean: '+str(mean_value)+', min: '+str(min_value)+', max: '+str(max_value))
	return mean_value , min_value, max_value

def top_values(G,item,value,nb_values=None):
	""" return the top values of data in graph G
		return a dataframe with first column(s) corresponding to node and edge
		with their associated value (sorted)
		item (string): 'edge' or 'node'
		value (string): name of the value attached to the nodes or edges of graph G
		nb_values (int or None, default=None): number of values to return, if nb_values=None return all values 
	"""
	if item=='edge':
		dfx = pd.DataFrame([ (n1,n2,data[value]) for n1,n2,data in G.edges(data=True)])
		dfx.columns = ['node1', 'node2', value]
	elif item=='node':
		dfx = pd.DataFrame([ (n,data[value]) for n,data in G.nodes(data=True)])
		dfx.columns = ['node', value]
	else:
		raise ValueError("only 'node' or 'edge' are accepted as item")
	dfx = dfx.sort_values([value],ascending=False)
	return dfx.head(nb_values)

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
		H.remove_nodes_from(nx.isolates(H))
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
	return G   

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
	"""
	#H = G.copy()
	H = G
	if node1 == node2: # if self-edge
		H.remove_edge(node1,node2)
	else:
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
	#H = normalize_weights(H,weight='weight')
	return H

def merge_nodes_respect_wiring(G, node1, node2, data=False):
	""" Partially merge two nodes node1 and node2 in the graph into one node 
		with id string node1+'_'+node2.
		The new node is exclusively connected to nodes on the paths that linked node1 and node2.
		node1 and node2 stays in the graph and keep only the connections of pths not passing between them.
		The connection between node1 and node2 is removed.

		if data=None, any data is ignored
		if data='node1' or 'node2' the new node inherit the data of the given node and
	"""
	import copy
	#H = G.copy()
	H = G
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
	text_ids = [x for x in H[node1][node2]['paths'].keys()]
	set1 = set(text_ids)
	#handle the outgoing connections
	edges_to_remove_suc = []
	for idx,suc in enumerate(H.successors(node2)):
		set2 = set([x for x in H[node2][suc]['paths'].keys()])
		common_elems = set1 & set2
		for elem in common_elems:
			list_of_positions = H[node1][node2]['paths'][elem]['word_positions']
			list_of_positions_next_edge = H[node2][suc]['paths'][elem]['word_positions']
			for word_position in list_of_positions:
				idx2 = find_next_idx(list_of_positions,list_of_positions_next_edge,word_position)
				#print('next edge ',idx2)
				if idx2>=0:
					#connect to new node
					add_connection_node(G,node_id,suc,elem,idx2)
					# disconnect from previous nodes
					edges_to_remove_suc.append((suc,elem,idx2))
				else: #TODO case where the next word idx is not just incremented by one
					pass
	# disconnect from the previous nodes
	#print('Nb of connections to remove: {}'.format(len(edges_to_remove)))
	for (node,text_id,word_pos) in edges_to_remove_suc:
		disconnect_node(G,node2,node,text_id,word_pos)

	#handle the ingoing connections
	edges_to_remove_pred = []
	for idx,pred in enumerate(H.predecessors(node1)):
		#print(pred,node1)
		set2 = set([x for x in H[pred][node1]['paths'].keys()])
		common_elems = set1 & set2
		for elem in common_elems:
			list_of_positions = H[node1][node2]['paths'][elem]['word_positions']
			list_of_positions_previous_edge = H[pred][node1]['paths'][elem]['word_positions']
			for word_position in list_of_positions:
				idx2 = find_previous_idx(list_of_positions,list_of_positions_previous_edge,word_position)
				#print('previous edge ',idx2)
				if idx2>=0:
					#connect to new node
					add_connection_node(G,pred,node_id,elem,idx2)
					# disconnect from previous nodes
					edges_to_remove_pred.append((pred,elem,idx2))
				else: 
					pass
	# disconnect from the previous nodes
	#print('Nb of connections to remove: {}'.format(len(edges_to_remove_pred)))
	for (node,text_id,word_pos) in edges_to_remove_pred:
		disconnect_node(G,node,node1,text_id,word_pos)
		
	# remove edge between node1 and node2
	H.remove_edge(node1,node2)
	# remove nodes if they are disconnected
	if not H.degree(node1):
		H.remove_node(node1)
	if not H.degree(node2):
		H.remove_node(node2)
	return H

def find_next_idx(list_of_positions,list_of_positions_next_edge,idx):
	list_of_positions.sort()
	idx_pos = list_of_positions.index(idx)
	if len(list_of_positions)>idx_pos+1:
		idx_stop = list_of_positions[idx_pos+1]
		allowed_pos = range(idx+1,idx_stop)
	else:
		allowed_pos = range(idx+1,idx+30) # TODO: get rid of the limitation to 30
	list_pos = [index for index in allowed_pos if index in list_of_positions_next_edge]
	if list_pos:
		return list_pos[0]
	else:
		return -1

def find_previous_idx(list_of_positions,list_of_positions_previous_edge,idx):
	list_of_positions.sort()
	idx_pos = list_of_positions.index(idx)
	if idx_pos>0:
		idx_stop = list_of_positions[idx_pos-1]
		allowed_pos = range(idx_stop,idx)
	else:
		allowed_pos = range(idx-30,idx) # TODO: get rid of the limitation to 30
	list_pos = [index for index in allowed_pos if index in list_of_positions_previous_edge]
	if list_pos:
		return list_pos[0]
	else:
		return -1

def add_connection_node(G,node1,node2,text_id,idx):
	#connect to new node
	if not G.has_edge(node1,node2):
		G.add_edge(node1,node2,paths={},weight=0)
		#H[pred][node_id] = {'paths':{},'weight':0}
	if text_id in G[node1][node2]['paths']:
		G[node1][node2]['paths'][text_id]['word_positions'].append(idx)
	else:
		G[node1][node2]['paths'][text_id] = {}
		G[node1][node2]['paths'][text_id]['word_positions'] = [idx]
	G[node1][node2]['weight'] += 1	

def disconnect_node(G,node1,node2,text_id,idx):
	edge = G[node1][node2]
	edge_idx_list = edge['paths'][text_id]['word_positions']
	idx_pos = edge_idx_list.index(idx)
	edge_idx_list.pop(idx_pos)
	edge['weight'] -=1
	if not len(edge_idx_list):
		del edge['paths'][text_id]
	if not len(edge['paths']):
		G.remove_edge(node1,node2)

def shrink_merge(G,max_nb_of_edges,start=0,step=0.001):
	""" Alternatively merge strong links and remove weak links to get a visualizable graph
	"""
	H = G.copy()
	#merge_strongly_connected_nodes(H,ratio=0.5,iterations=1)
	threshold = start
	while H.size() > max_nb_of_edges:
		print(threshold)
		H = remove_weak_links(H,threshold)
		H.remove_nodes_from(nx.isolates(H))
		H = normalize_weights(H,weight='weight')
		merge_strongly_connected_nodes(H,ratio=0.5,iterations=1)
		threshold +=step
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
