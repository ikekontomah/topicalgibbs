#!/usr/bin/env python


import random
import numpy as np
from copy import copy
from itertools import izip, count
from bart_utils import empty,Tree, logsumexp,softmax,check_if_zero,get_children_id


class Particle(Tree):
	def _init__(self,train_ids =np.arange(0,dtype='int'),param= empty(),settings=empty(),cache_tmp = {}):
		Tree.__init(self,train_ids,param,settings,cache_tmp)
		self.ancestry = []
		self.processed_nodes_iterations = []
		self.grow_nodes_iterations = []
		self.log_sis_ratio_d ={}
		if cache_tmp:
			self.do_not_grow = False
			self.grow_nodes = [0]

	def process_node_id(self,data,param , settings, cache, id_of_node):
		if self.do_not_split[id_of_node]:
			log_sis_ratio =0.0
		else:
			log_psplit = np.log(self.compute_psplit(id_of_node,param))
			train_ids = self.train_ids[id_of_node]
			left_node,right_node = get_children_id(id_of_node)
			if setings.verbose >= 4:
				print('Train_ids for this node =%s' % train_ids)
			(dont_split_node_id,id_feat_chosen,chosen_split,split_global_idx,log_sis_ratio,logprior_nodeid,training_ids_left,training_ids_right,cache_tmp,loglikelihood_left,loglikelihood_right) =\
			self.prior_proposal(data,param,settings,cache,id_of_node,train_ids,log_psplit)
			if dont_split_node_id:
				self.do_not_split[id_of_node] = True
			else:
				self.update_left_right_statistics(cache_tmp,id_of_node,logprior_nodeid,training_ids_left,training_ids_right,loglikelihood_left,loglikelihood_right,id_feat_chosen,chosen_split,split_global_idx,settings,param,data,cache)
				self.grow_nodes.append(left_node)
				self.grow_nodes.append(right_node)
		return (log_sis_ratio)

	def grow_next(self,data,param,settings,cache):
		do_not_grow = True
		log_sis_ratio = 0.0
		processed_nodes = []
		if not self.grow_nodes:
			if settings.verbose >= 2:
				print('The leaves cant be grown anymore: Current depth=%3d , skipping grow_next' % self.depth)
		else:
			while True:
				remove_position = 0
				id_of_node = self.grow_nodes.pop(remove_position)
				processed_nodes.append(id_of_node)
				do_not_grow = do_not_grow and self.do_not_split[id_of_node]
				if self.do_not_split[id_of_node]:
					if settings.verbose >= 3:
						print('Skip the split at the given node_id %3d' % id_of_node)

					if not self.grow_nodes:
						break
				else:
					log_sis_ratio += self.process_node_id(data,param ,settings, cache, id_of_node)
					break
			self.loglik_current = self.compute_loglik()
		self.log_sis_ratio = log_sis_ratio
		self.do_not_grow = do_not_grow
		if processed_nodes:
			self.processed_nodes_iterations.append(processed_nodes)

	def check_nodes_processed_itr(self,settings):
		temporal_set = set([])
		for nodes_to_use in self.processed_nodes_iterations:
			for a_node in nodes_to_use:
				if a_node in temporal_set:
					print('node = %s present multiple times in processed_nodes_iterations = %s' % (a_node,self.processed_nodes_iterations))
					raise Exception
				else:
					temporal_set.add(a_node)

def update_particle_weights(particles,log_weights,settings):
	for n,p in enumerate(particles):
		if settings.verbose >=2:
			print('pid = %5d, log_sis_ratio =%f' % (n,p.log_sis_ratio))
		log_weights[n] +=p.log_sis_ratio
	norm_of_weights = softmax(log_weights) # normalize the weights
	ess = 1./ np.sum(norm_of_weights**2)/ settings.n_particles
	log_pd = logsumexp(log_weights)
	return (log_pd,ess,log_weights,norm_of_weights)

def resample(particles,log_weights,settings,log_pd,ess,norm_of_weights,tree_pg):
	if ess < settings.ess_threshold:
		if tree_pg:
			pid_list = resample_pids_basic(settings,settings.n_particles-1,norm_of_weights)
			random.shuffle(pid_list)
			pid_list.insert(0,0)
		else:
			pid_list = resample_pids_basic(settings,settings.n_particles,norm_of_weights)
		log_weights = np.ones(settings.n_particles)*(log_pd-np.log(settings.n_particles))
	else:
		pid_list = range(settings.n_particles)
	if settings.verbose >= 2:
		print('ess = %s,ess_threshold = %s' % (ess,settings.ess_threshold))
		print('New particle ids = ')
		print(pid_list)
	op = create_new_particles(particles, pid_list,settings)
	for pid, p in izip(pid_list,op):
		p.ancestry.append(pid)
	return (op,log_weights)


def resample_pids_basic(settings,n_particles,prob):
	if settings.resample == 'multinomial':
		pid_list = sample_multinomial_numpy(n_particles,prob)

	elif settings.resample == 'systematic':
		pid_list = systeatic_sample(n_particles,prob)
	return pid_list

def sample_multinomial_numpy(n_particles,prob):
	indices = np.random.multinomial(n_particles,prob,size = 1)
	pid_list_to_use = [pid for pid,counting in enumerate(indices.flat) for n in range(counting)]
	return pid_list_to_use

def create_new_particles(particles,pid_list,settings):
	allocated_list =set([])
	op =[]
	for ele,pid in enumerate(pid_list):
		if pid not in allocated_list:
			op.append(particles[pid])
		else:
			op.append(copy_particle(particles[pid],settings))
		allocated_list.add(pid)
	return op


def copy_particle(p,settings):

	op = Particle()
	op.leaf_nodes = p.leaf_nodes[:]
	op.non_leaf_nodes = p.non_leaf_nodes[:]
	op.ancestry = p.ancestry[:]
	op.processed_nodes_iterations = [x[:] for x in p.processed_nodes_iterations]
	op.grow_nodes = p.grow_nodes[:]
	op.grow_nodes_iterations = [x[:] for x in p.grow_nodes_iterations]
	op.do_not_split = p.do_not_split.copy()
	op.log_sis_ratio_d = p.log_sis_ratio_d.copy()
	op.sum_y = p.sum_y.copy()
	op.sum_y2 = p.sum_y2.copy()
	op.n_points = p.n_points.copy()
	op.param_n = p.param_n.copy()
	op.train_ids = p.train_ids.copy()
	op.node_info = p.node_info.copy()
	op.loglik = p.loglik.copy()

	op.depth = copy(p.depth)
	op.do_not_grow = copy(p.do_not_grow)
	op.loglik_current = copy(p.loglik_current)
	return op

def systematic_sample(n,prob):
	assert(n== len(prob))
	assert(abs(np.sum(prob)-1) < 1e-10)
	cummulative_probability = np.cumsum(prob)
	ele_prob = np.random.rand(1)/float(n)
	ele = 0
	indices = []
	while True:
		while ele_prob > cummulative_probability[ele]:
			ele += 1
		indices.append(ele)
		ele_prob += 1/float(n)
		if ele_prob > 1:
			break
	return indices

def init_particles(data,settings,param,cache_tmp):
	particles = [Particle(np.arange(data['n_train']),param,settings,cache_tmp) for n in range(settings.n_particles)]
	log_weights = np.array([p.loglik[0] for p in particles])-np.log(settings.n_particles)
	return (particles,weights)

def grow_next_pg(p,tree_pg,iterations,settings):
	p.log_sis_ratio = 0.
	p.do_not_grow = False
	p.grow_nodes = []
	try:
		processed_nodes = tree_pg.processed_nodes_iterations[iterations]
		p.processed_nodes_iterations.append(processed_nodes[:])
		for id_of_node in processed_nodes[:-1]:
			assert(tree_pg.do_not_split[id_of_node])
			p.do_not_split[id_of_node] = True
		id_of_node = processed_nodes[-1]
		if id_of_node in tree_pg.node_info:
			left_node,right_node = get_children_id(id_of_node)
			log_sis_ratio_loglik_new = tree_pg.loglik[left_node] + tree_pg.loglik[right_node] - tree_pg.loglik[id_of_node]
			try:
				log_sis_ratio_loglik_old , log_sis_ratio_prior = tree_pg.log_sis_ratio_d[id_of_node]
			except KeyError:
				print('tree_pg: node_info =%s, log_sis_ratio_d = %s' % (tree_pg.node_info,tree_pg.log_sis_ratio_d))
				raise KeyError
			if settings.verbose >= 2:
				print('log_sis_ratio_loglik_old =%s' % log_sis_ratio_loglik_old)
				print('log_sis_ratio_loglik_new = %s' % log_sis_ratio_loglik_new)
			p.log_sis_ratio = log_sis_ratio_loglik_new + log_sis_ratio_prior
			tree_pg.log_sis_ratio_d[id_of_node] = (log_sis_ratio_loglik_new,log_sis_ratio_prior)
			p.log_sis_ratio_d[id_of_node] = tree_pg.log_sis_ratio_d[id_of_node]
			p.non_leaf_nodes.append(id_of_node)
			try:
				p.leaf_nodes.remove(id_of_node)
			except ValueError:
				print('Warning !!! Unable to remove id_of_node =%s from leaf_nodes = %s' %(id_of_node,p.leaf_nodes))
				pass
			p.leaf_nodes.append(left_node)
			p.leaf_nodes.append(right_node)
			p.node_info[id_of_node] = tree_pg.node_info[id_of_node]
			p.logprior[id_of_node] = tree_pg.logprior[id_of_node]
			for child_node_id in [left_node,right_node]:
				p.do_not_split[child_node_id] = False
				p.loglik[child_node_id] = tree_pg.loglik[child_node_id]
				p.logprior[child_node_id] = tree_pg.logprior[child_node_id]
				p.train_ids[child_node_id] = tree_pg.train_ids[child_node_id]
				p.sum_y[child_node_id] = tree_pg.sum_y[child_node_id]
				p.sum_y2[child_node_id] = tree_pg.sum_y2[child_node_id]
				p.param_n[child_node_id] = tree_pg.param_n[child_node_id]
				p.n_points[child_node_id] = tree_pg.n_points[child_node_id]
		if settings.verbose >= 2:
			print('p.leaf_nodes = %s' % p.leaf_nodes)
			print('p.non_leaf_nodes =%s' % p.non_leaf_nodes)
			print('p.node_info.keys() = %s' % sorted(p.node_info.keys()))
		try:
			p.grow_nodes = tree_pg.grow_nodes_iterations[iterations+1]
			p.log_sis_ratio_d = tree_pg.log_sis_ratio_d
			p.depth = tree_pg.depth
		except IndexError:
			p.do_not_grow = True
	except IndexError:
		p.do_not_grow = True

def run_smc(particles, data, settings, param, log_weights,cache,tree_pg = None):
	if settings.verbose >= 2:
		print('Conditioned tree:')
		tree_pg.print_tree()
	iteration = 0
	while True:
		if settings.verbose >= 2:
			print('\n')
			print('*'*80)
			print('Current iteration = %3d' % iteration)
			print('*'*80)
		if iteration != 0:
			if settings.verbose >= 1:
				print('Iteration = %3d , logp(y|x) = %.2f, ess/n_particles = %f' % (iteration,log_pd,ess))
			(particles, log_weights) = resample(particles,log_weights,settings,log_pd,ess,norm_of_weights,tree_pg)
		for pid, p in enumerate(particles):
			if settings.verbose >= 2:
				print('Current particle = %3d' % pid)
				print('Grow nodes =%s' % p.grow_nodes)
				print('Leaf nodes =%s,non_leaf_nodes= %s' % (p.leaf_nodes,p.non_leaf_nodes))
			if p.grow_nodes:
				p.grow_nodes_iterations.append(p.grow_nodes[:])
			if tree_pg and (pid ==0):
				if settings.verbose >= 2 and iteration == 0:
					for s in ['leaf_nodes','non_leaf_nodes','grow_nodes_iterations','ancestry','processed_nodes_iterations']:
						print('p.%s = %s' % (s,getattr(p,s)))
					grow_next_pg(p,tree_pg,iteration,settings)
			else:
				p.grow_next(data,param,settings,cache)
			p.update_depth()
			if settings.verbose >= 2:
				print('processed_nodes_iterations for particle =%s' % p.processed_nodes_iterations)
				print('grow_nodes (after running grow_next) (NOT updated for conditioned tree_pg)=%s'% p.grow_nodes)
				print('leaf_nodes = %s,non_leaf_nodes=%s' % (p.leaf_nodes,p.non_leaf_nodes))
				print('processed_nodes_iterations for particles (after running upate_particle weights)=%s' %p.processed_nodes_iterations)
				print('checking processed_nodes_iterations...')

			
        (log_pd,ess,log_weights,norm_of_weights) = update_particle_weights(particles,log_weights,settings)
		if settings.verbose >= 2:
			print('log_weights =%s' % log_weights)
		if check_do_not_grow(particles):
			if settings.verbose >= 1:
				print('None of the particles can be grown any further; breaking out')
		iteration += 1
	if (settings.debug == 1) and tree_pg:
		for pid,p in enumerate(particles):
			if settings.verbose >= 2:
				print('Checking pid =%s' % pid)
			p.check_nodes_processed_itr(settings)
		if settings.verbose >= 2:
			print('Check if tree_pg did the right thing')
			print('nodes_processed_iterations (original,new): \n%s\n%s' %(tree_pg.processed_nodes_iterations,particles[0].processed_nodes_iterations))
			print('leaf_nodes (orig, new): \n%s\n%s' % (tree_pg.leaf_nodes,particles[0].leaf_nodes))
			print('non_leaf_nodes (original,new):\n%s\n%s' % (tree_pg.non_leaf_nodes,particles[0].non_leaf_nodes))
			print('grow_nodes_iterations (original,new):\n%s\n%s' % (tree_pg.grow_nodes_iterations,particles[0].grow_nodes_iterations))
		assert particles[0].leaf_nodes == tree_pg.leaf_nodes
		assert particles[0].non_leaf_nodes == tree_pg.non_leaf_nodes
		assert particles[0].grow_nodes_iterations == tree_pg.grow_nodes_iterations
	return (particles,ess,log_weights,log_pd)


def init_run_smc(data,settings,param,cache,cache_tmp,tree_pg = None):
	particles,log_weights = init_particles(data,settings,param,cache_tmp)
	(particles,ess,log_weights,log_pd) = run_smc(particles,data,settings,param,log_weights,cache,tree_pg)
	return (particles,log_pd,log_weights)



def check_do_not_grow(particles):
	do_not_grow = True
	for part in particles:
		do_not_grow = do_not_grow and p.do_not_grow
	return do_not_grow















