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

def merge_nodes_respect_wiring_deprecated(G, node1, node2, data=False):
	""" Partially merge two nodes node1 and node2 in the graph into one node 
		with id string node1+'_'+node2.
		The new node is exclusively connected to nodes on the paths that linked node1 and node2.
		node1 and node2 stays in the graph and keep only the connections of pths not passing between them.
		The connection between node1 and node2 is removed.

		if data=None, any data is ignored
		if data='node1' or 'node2' the new node inherit the data of the given node and
	"""
	import copy
	if not (node1 in G):
		print("Warning: node '{}' not in the graph".format(node1))
	elif not (node2 in G):
		print("Warning: node '{}' not in the graph".format(node2))
	else:
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

def merge_nodes_respect_wiring(G, node1, node2, data=False):
	""" Partially merge two nodes node1 and node2 in the graph into one node 
		with id string node1+'_'+node2.
		The new node is exclusively connected to nodes on the paths that linked node1 and node2.
		node1 and node2 stays in the graph and keep only the connections of paths not passing between them.
		The connection between node1 and node2 is removed.
		If all the connections were only between node1 and node2, removing the edge produce 2 isolated nodes,
		then they are removed from the graph.

		Additionally, the node resulting from a merge get the info of the paths, saved as a node property. 
		This provides a way to find the texts containing the merged words.

		if data=None, any data is ignored
		if data='node1' or 'node2' the new node inherit the data of the given node
		if data='auto', the new node inherit the data of the node having some data,
						if both have data, the node having the largest degree
	"""
	import copy
	if not (node1 in G):
		print("Warning: node '{}' not in the graph".format(node1))
	elif not (node2 in G):
		print("Warning: node '{}' not in the graph".format(node2))
	else:
		if not G.has_edge(node1,node2):
			print("Warning: no edge found between nodes {} and {}".format(node1,node2))
		else:
			#H = G.copy()
			H = G
			# create the new node
			node_id = node1+'_'+node2
			#node_imbed_links = copy.deepcopy(H[node1][node2]['paths'])
			if data == False:
				H.add_node(node_id)
			elif data == node1:
				node_data = copy.deepcopy(H.node[node1])
				H.add_node(node_id, node_data)
			elif data == node2:
				node_data = copy.deepcopy(H.node[node2])
				H.add_node(node_id, node_data)
			elif data =='auto':
				degree1 = len(G[node1])
				degree2 = len(G[node2])
				if degree1 > degree2 and H.node[node1]:
					node_data = copy.deepcopy(H.node[node1])
					H.add_node(node_id, node_data)
				else:
					node_data = copy.deepcopy(H.node[node2])
					H.add_node(node_id, node_data)
			else:
				raise ValueError("data only accept False, auto or the id of one of the nodes")
			# connect it to the rest

			#handle outgoing connections
			edges_to_remove = copy_links(G,node1,node2,node_id,direction='out')
			# disconnect from the previous nodes
			for (node,text_id,word_pos) in edges_to_remove:
				disconnect_node(G,node2,node,text_id,word_pos)

			#handle the ingoing connections
			edges_to_remove = copy_links(G,node1,node2,node_id,direction='in')
			# disconnect from the previous nodes
			paths_dic = {}
			for (node,text_id,word_pos) in edges_to_remove:
				disconnect_node(G,node,node1,text_id,word_pos)

			# Compare the paths saved on the node to the paths on the edges
			# and update the paths:
			# some paths get separated and have to be removed
			# We want to keep the word position of the first word on the merged words
			# The solution is to save the text and position from the edge to the merged node
			# We have to handle the case where a node to be merged is already the product
			# of a merge
			if 'paths' in H.node[node1]: # if there are paths, compare them to the edge ones
				# copy the dict 
				# since we want to iterate on a copy of it and modify 2 other copies
				node_paths = copy.deepcopy(H.node[node1]['paths'])
				H.node[node_id]['paths'] = copy.deepcopy(node_paths)
				edge_paths = H[node1][node2]['paths']
				for text_id in node_paths.keys():
					if text_id in edge_paths:
						for idx in node_paths[text_id]['word_positions']:
							idx2 = find_next_idx(node_paths[text_id]['word_positions'],
								edge_paths[text_id]['word_positions'],idx)
							if idx2==-1: # if there is no path corresponding to the node path in the edge paths,
								H.node[node_id]['paths'][text_id]['word_positions'].pop(
									H.node[node_id]['paths'][text_id]['word_positions'].index(idx))
							else:
								H.node[node1]['paths'][text_id]['word_positions'].pop(
									H.node[node1]['paths'][text_id]['word_positions'].index(idx))
								if not H.node[node1]['paths'][text_id]['word_positions']:
									del H.node[node1]['paths'][text_id]
					else:
						del H.node[node_id]['paths'][text_id]
			else:
				H.node[node_id]['paths']=H[node1][node2]['paths']

			# handle the case where data about connections are stored on node2
			if 'paths' in H.node[node2]:
				node_paths = copy.deepcopy(H.node[node2]['paths'])
				edge_paths = H[node1][node2]['paths']
				for text_id in edge_paths.keys():
					if text_id in node_paths:
						for idx in edge_paths[text_id]['word_positions']:
							idx2 = find_next_idx(edge_paths[text_id]['word_positions'],
								node_paths[text_id]['word_positions'],idx)
							if idx2>=0: # if there is a path corresponding to the edge path in the node2 paths,
								H.node[node2]['paths'][text_id]['word_positions'].pop(
									H.node[node2]['paths'][text_id]['word_positions'].index(idx2))
								if not H.node[node2]['paths'][text_id]['word_positions']:
									del H.node[node2]['paths'][text_id]



			# remove edge between node1 and node2
			H.remove_edge(node1,node2)
			# remove nodes if they are disconnected
			if not H.degree(node1):
				H.remove_node(node1)
			if node2 in H and not H.degree(node2): # first check for the case node1==node2, then the degree
				H.remove_node(node2)
	return H

def copy_links(G,node1,node2,node3,direction):
	""" Follow the links between node1 and node2 of graph G to their neighbors, according the direction.
		if direction='out', the out edges from node2 are followed and its neighbors are connected
		to node3
		if direction='in', the in edges to node1 are followed back and the connected neighbors are
		connected to node3
		return the list of edges that have been copied
	"""
	if direction=='out':
		edge_list = G.successors(node2)
		source_node = node2
	elif direction == 'in':
		edge_list = G.predecessors(node1)
		target_node = node1
	else:
		raise ValueError("direction can only be 'in' or 'out'.")
	
	text_ids = [x for x in G[node1][node2]['paths'].keys()]
	set1 = set(text_ids)
	edges_copied = []
	for idx,n_neighbors in enumerate(edge_list):
		if not (n_neighbors==node1 or n_neighbors==node2): #avoid selfloops and reverse direction
			if direction=='out':
				target_node = n_neighbors
			else:
				source_node = n_neighbors
			#print(pred,node1)
			set2 = set([x for x in G[source_node][target_node]['paths'].keys()])
			common_elems = set1 & set2
			for text_id in common_elems:
				list_of_positions = G[node1][node2]['paths'][text_id]['word_positions']
				list_of_positions_sourcetarget = G[source_node][target_node]['paths'][text_id]['word_positions']
				for word_position in list_of_positions:
					if direction=='out':
						idx2 = find_next_idx(list_of_positions,list_of_positions_sourcetarget,word_position)
						source_node2,target_node2 = node3,target_node
					else:
						idx2 = find_previous_idx(list_of_positions,list_of_positions_sourcetarget,word_position)
						source_node2,target_node2 = source_node,node3
					#print('previous edge ',idx2)
					if idx2>=0:
						#connect to new node
						add_connection_node(G,source_node2,target_node2,text_id,idx2)
						# disconnect from previous nodes
						edges_copied.append((n_neighbors,text_id,idx2))
	return edges_copied


def find_next_idx(list_of_positions,list_of_positions_next_edge,idx):
	""" from two lists and one value of the first list, return the value correxponding
		to this index incremented by an unknown amount but: 
		it must be less than the next value in the first list (ordered)
		it must be less than some fixed value (30) 
	"""
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
	""" Same as find_next_idx but find if an index exist previously to the given idx.
		See find_next_idx.
	"""
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
	""" Add a connection between node1 and node2 with edge properties:
		a dict with key : text_id, and value a new dict.
		This latter dic has key 'word_positions' and value: [idx] (list).
		If this dic and key already exist, idx is appended to the list.
	"""
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
	""" remove a connection between node1 and node2 
		with text id 'text_id' and word position 'idx'.
	"""
	edge = G[node1][node2]
	edge_idx_list = edge['paths'][text_id]['word_positions']
	idx_pos = edge_idx_list.index(idx)
	edge_idx_list.pop(idx_pos)
	edge['weight'] -=1
	if not len(edge_idx_list):
		del edge['paths'][text_id]
	if not len(edge['paths']):
		G.remove_edge(node1,node2)


def merge_strongly_connected_nodes_fast(G,min_weight,max_iter=1000):
	""" Iteratively merge the strongest connections in a graph while respecting the wiring.
		The process is done until the strongest connection has a weight of min_weight 
		or max_iter has been reached. 
	"""
	# Create a lightweight copy of the graph, without the smallest links
	print('Copying the graph...')
	H = G.copy()
	threshold = min_weight
	print('Shrinking the copy...')
	H = remove_weak_links(H,threshold,weight='weight')
	H.remove_nodes_from(nx.isolates(H))
	# Search for the strongest connections on the small graph
	# and merge them both on the small and full graphs
	nb_of_edges = H.size()
	for i in range(max_iter):
		top_table = top_values(H,'edge','weight',nb_values=1)
		#if not i%20:
		node1,node2,weight = top_table.iloc[0].node1,top_table.iloc[0].node2,top_table.iloc[0].weight
		print('Iter {}.== {} {} ==. Weight {} (estimated nb of iter.: {}).'.format(i,node1,node2,weight,nb_of_edges))
		H = merge_nodes_respect_wiring(H, node1, node2, data=False)
		G = merge_nodes_respect_wiring(G, node1, node2, data=False)
		if weight<min_weight:
			break
	return G


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
# Create the graph of document

def doc_graph(G):
	""" Create a graph of documents from the graph of words.

	G: graph of words
	Return
	G_doc graph of documents (indexed by their document Id).

	"""
	G_doc = nx.Graph()
	for node,data in G.nodes(data=True):
		if 'paths' in data.keys():
			list_of_docs = data['paths'].keys()
			if len(list_of_docs)>1:
				for pair in itertools.combinations(list_of_docs, 2):
					if G_doc.has_edge(*pair):
						G_doc[pair[0]][pair[1]]['weight']+=1
					else:
						G_doc.add_edge(pair[0],pair[1],weight=1)
	return G_doc

################################################################
# Community detection and classification

def find_communities(G):
	""" Community detection on graph G

	Uses the module 'community' to find communities in the graph.
	The nodes are labelled inplace.
	Return
	G: where all the nodes have been labelled in a community
	culsterDic: dictionary of nodes id as key and cluster id as value

	"""
	import community
	#first compute the best partition
	clusterDic = community.best_partition(G)
	nb_communities = len(set(clusterDic.values()))
	print('Nb of communities found: {}'.format(nb_communities))
	nx.set_node_attributes(G,'cluster',clusterDic)
	return G,clusterDic

def get_filenames_in_clusters(clusterDic,df_filenames):
	""" Find the filenames contained in each communities
	clusterDic: dictionary with documnent id as key and cluster id as value
	df_filenames: pandas dataframe indexed by the document ids and with a column 'filename'
	return:
	cluster_dic_name, a dictionary with cluster as key and list of filenames as value
	
	"""
	# First step:
	# Create a dict with clusters id as key and the list of document ids as value
	nb_communities = len(set(clusterDic.values()))
	cluster_dic = {}
	for cluster_i in range(nb_communities):
		doc_list = [idx for idx,value in clusterDic.items() if value==cluster_i]
		cluster_dic[cluster_i] = doc_list
	# Second step:
	# Replace the document id with the filename of the document
	cluster_dic_name = {}
	for key in cluster_dic.keys():
		list_of_names = []
		for idx in cluster_dic[key]:
			list_of_names.append(df_filenames.loc[int(idx),'filename'])
		cluster_dic_name[key]=list_of_names
	return cluster_dic_name 

def extract_cluster_as_subgraph(G,cluster_id):
	""" Extract the subgraph with nodes belonging to community cluster_id
	
	Make a copy of the graph, return a subgraph of G.
	Return a graph

	"""
	"""
	G_c = G.copy()
	for node,data in G_c.nodes(data=True):
		if not data['cluster']==cluster_id:
			G_c.remove_node(node)
	print('Nb of edges of the subgraph: {}, nb of nodes: {}'.format(G_c.size(),len(G_c.nodes())))
	return G_c
	"""
	G_c = nx.Graph()
	for node1,node2,data in G.edges(data=True):
		if G.node[node1]['cluster']==cluster_id and G.node[node2]['cluster']==cluster_id:
			G_c.add_edge(node1,node2,data)
			G_c.node[node1] = G.node[node1]
			G_c.node[node2] = G.node[node2]
	print('Nb of edges of the subgraph: {}, nb of nodes: {}'.format(G_c.size(),len(G_c.nodes())))
	return G_c	

def cluster_graph(G,min_cluster_size):
	""" Separate the graph G into communities that have a specified scale (size)

	Find communities in graph G and recusively communities within communities,
	until the communities cannot be separated further or have reached a size smaller
	than min_cluster_size
	return:
	list of subgraphs, each one corresponding to a community

	"""
	subgraph_list = []
	graph = G.copy()
	if len(graph.nodes())>min_cluster_size:
		graph,clusterDic = find_communities(graph)
		nb_communities = len(set(clusterDic.values()))
		if nb_communities > 1:
			for c_i in range(nb_communities): 
				G_sub = extract_cluster_as_subgraph(graph,cluster_id=c_i)
				[subgraph_list.append(item) for item in cluster_graph(G_sub,min_cluster_size)]
		else:
			subgraph_list.append(graph)
	else:
		subgraph_list.append(graph)
	return subgraph_list

def clusters_info(subgraph_list):
	import numpy as np
	print('Nb of communities:',len(subgraph_list))
	list_nb_nodes = [len(graph.nodes()) for graph in subgraph_list]
	print('Community mean size: {:.2f}, min size: {}, max size: {}'.format(np.mean(list_nb_nodes),
		np.min(list_nb_nodes),np.max(list_nb_nodes)))


def subgraphs_to_filenames(list_of_graphs,df_of_filenames,density=False):
	""" Return the list of filenames associated to each graph in the list_of_graphs.
		
		for each graph in the list of graphs, associate the nodes id to their filename
		using df_of_filenames.
		Return:
		list of list containing the filenames for each community (graph)
		if density==True, append at the end of each filename list the density of the subgraph

	"""
	cluster_name_list = []
	for graph in list_of_graphs:
		subgraph_names_list = []
		for node in graph:
			subgraph_names_list.append(df_of_filenames.loc[int(node),'filename'])
		cluster_name_list.append(subgraph_names_list)
		if density == True:
			subgraph_names_list.append(nx.density(graph))
	return cluster_name_list

def output_filename_classification(cluster_name_list,csv_filename):
	""" Save the classification in a csv file
	
	Each column correspond to a cluster.
	Along the columns are the filenames classified in the corresponding cluster.
	Return the dataframe.

	"""
	# Create a dataframe from filename lists
	clusters_table = pd.DataFrame()
	for idx,name_list in enumerate(cluster_name_list):
		df1=pd.DataFrame(name_list)
		clusters_table = pd.concat([clusters_table,df1], ignore_index=True, axis=1)
	print('Save to file {}'.format(csv_filename))
	clusters_table.to_csv(csv_filename)
	return clusters_table

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
