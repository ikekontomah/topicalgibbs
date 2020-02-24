#!/usr/bin/env python

import sys
import optparse
import math
import cPickle as pickle
import numpy as np
import pprint as ppt
from scipy.special import gammaln , digamma
from copy import copy
from bart_utils import Tree,process_command_line,get_filename_bart,compute_metrics_regression,precompute,\
     update_cache_tmp,compute_gamma_loglik, compute_normal_loglik,init_performance_storage,get_k_data_names,\
     store_every_iteration,load_data
import random
from pg import Particle
from treemcmc import TreeMCMC, init_tree_mcmc, run_mcmc_single_tree
import time
np.seterr(divide='ignore')
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=np.inf)



class BART(object):
    def __init__(self,data,param,settings,cache,cache_tmp):
        self.trees =[]
        self.pmcmc_objects=[]
        self.predicted_value_mat_train = np.zeros((data['n_train'],settings.m_bart))
        self.update_predicted_value_sum()
        for ele in range(settings.m_bart):
            p,predicted_tmp,pmcmc = init_tree_mcmc(data,settings,param,cache,cache_tmp)
            sample_param(p,settings,param,False)
            self.trees.append(p)
            self.pmcmc_objects.append(pmcmc)
            self.update_predicted_value(ele,data,param,settings)
        self.lambda_logprior = compute_gamma_loglik(param.lambda_bart,param.alpha_bart,param.beta_bart)

    def update_predicted_value(self,ele,data_to_use,param,settings):
        self.trees[ele].gen_rules_tree()
        temporary = self.trees[ele].predict_real_val_fast(data_to_use['x_train'],param,settings)
        if settings.debug == 1:
            print('currently we have data[y_train] as = %s' % data['y_train'])
            print('The tree has current predictions = %s' % tmp)
        self.predicted_value_sum_train += temporary
        self.predicted_value_sum_train -= self.predicted_value_mat_train[:,ele]

        self.predicted_value_mat_train[:,ele] = temporary
        if settings.debug == 1:
            print('current data[y_train_original] = %s' % data['y_train_orig'])
            print('predictions from current bart are = %s' % self.predicted_value_sum_train)

    def update_predicted_value_sum(self):
        self.predicted_value_sum_train = self.predicted_value_mat_train.sum(1)

    def compute_mse_within_and_without(self,ele,data,settings):
        predicted_tree = self.predicted_value_mat_train[:,ele]
        predicted_tree_variance = np.var(predicted_tree)
        prediction_without_tree = self.predicted_value_sum_train - self.predicted_value_mat_train[:,ele]
        mse_tree = compute_mse(data['y_train_orig'],predicted_tree)
        mse_without_tree = compute_mse(data['y_train_orig'],prediction_without_tree)
        return (mse_tree,mse_without_tree,predicted_tree_variance)

    def update_residual(self,ele,data):
        residual_value = data['y_train_orig']-self.predicted_value_sum_train + self.predicted_value_mat_train[:,ele]
        data['y_train'] = residual_value

    def sample_labels(self,data,settings,param):
        data['y_train_orig'] = self.predicted_value_sum_train + np.random.randn(data['n_train'])/math.sqrt(param.lambda_bart)

    def predict(self,x,y,param,settings):
        predicted_probability = np.zeros(x.shape[0])
        predicted_value = np.zeros(x.shape[0])
        log_constant    = 0.5*math.log(param.lambda_bart)-0.5*math.log(2*math.pi)
        for ele in range(settings.m_bart):
            exec(self.trees[ele].rules)
            predicted_value += self.trees[ele].pred_val_n[leaf_id]
        predicted_probability = np.exp(-0.5*param.lambda_bart*((y-predicted_value)**2)+log_constant)
        return {'pred_mean':predicted_value,'pred_prob':predicted_probability}

    def predict_training(self,data,param,settings):
        predicted_value = self.predicted_value_sum_train.copy()
        log_constant = 0.5*math.log(param.lambda_bart) - 0.5*math.log(2*math.pi)
        log_likelihood = -0.5*param.lambda_bart*((data['y_train_orig']-predicted_value)**2+ log_constant)
        predicted_probability = np.exp(log_likelihood)
        return {'pred_mean':predicted_value,'pred_prob':predicted_probability}
    
    def sample_lambda_bart(self,param,data,settings):
        lambda_alpha = param.alpha_bart + 0.5*data['n_train']
        lambda_beta = param.beta_bart + 0.5*np.sum((data['y_train_orig']-self.predicted_value_sum_train)**2)
        param.lambda_bart = float(np.random.gamma(lambda_alpha,1.0/lambda_beta, 1))
        param.log_lambda_bart = math.log(param.lambda_bart)
        self.lambda_logprior = compute_gamma_loglik(param.lambda_bart,param.alpha_bart,param.beta_bart)

    def compute_train_mse(self,data,settings):
        mse_training = compute_mse(data['y_train_orig'],self.predicted_value_sum_train)
        if settings.verbose >= 1:
            print('mse train = %.3f' %(mse_training))
        return mse_training

    def compute_train_mse_original(self,data,settings,param):
        return compute_mse(data['f_train'],self.predicted_value_sum_train)
        
    def compute_train_loglikelihood(self,data,settings,param):
        mse_training = compute_mse(data['y_train_orig'], self.predicted_value_sum_train)
        loglikelihood_training = 0.5*data['n_train']*(math.log(param.lambda_bart)-math.log(2*math.pi)-param.lambda_bart*mse_training)
        return (loglikelihood_training,mse_training)

    
def compute_mse(true_value,predicted_value):
    return np.mean((true_value-predicted_value)**2)

def sample_param(p,settings,param,set_to_mean=False):
    p.pred_val_n = np.inf* np.ones(max(p.leaf_nodes)+1)
    p.pred_val_logprior = 0.
    for id_of_node in p.leaf_nodes:
        mu_of_mean_post,mu_prec_post = p.param_n[id_of_node]
        if set_to_mean:
            p.pred_val_n[id_of_node] = 0. + mu_of_mean_post
        else:
            p.pred_val_n[id_of_node] = float(np.random.randn(1)/np.sqrt(mu_prec_post))+ mu_of_mean_post
        p.pred_val_logprior += compute_normal_loglik(p.pred_val_n[id_of_node],param.mu_mean,param.mu_prec)
        

def center_labels(data , settings):
    data['y_train_mean'] = np.mean(data['y_train'])
    data['y_train']      -= data['y_train_mean']
    data['y_test']       -= data['y_train_mean']


def backup_target(data,settings):
    data['y_train_orig'] = data['y_train'].copy()
    data['y_test_orig']  = data['y_test'].copy()

def main():
    settings = process_command_line()
    print('Current Settings:')
    ppt.pprint(vars(settings))

    np.random.seed(settings.init_id*1000)
    random.seed(settings.init_id*1000)

    print("Loading data....")
    data = load_data(settings)
    print("Dating loading completed")

    if settings.center_y:
        print('center_y = True; center the y variables at mean(data[y_train])')
        center_labels(data, settings)
    backup_target(data,settings)

    time_start = time.clock()
    param , cache , cache_tmp = precompute(data,settings)
    bart = BART(data,param,settings,cache,cache_tmp)
    time_initialization = time.clock() - time_start
    
    mcmc_stats = np.zeros((settings.m_bart,settings.n_iterations,10))
    mcmc_stats_bart = np.zeros((settings.n_iterations, 10))
    mcmc_stats_bart_desc = ['loglik','logprior','logprob','mean_depth','mean num_leaves','mean num_nonleaves','mean change','mse_train','lambda_bart','time_itr']
    mcmc_counts = None
    mcmc_tree_predictions = init_performance_storage(data, settings)

    burn_in_number = 0
    assert burn_in_number == 0

    init_time = time.clock()
    init_time_run_average = time.clock()
    iteration_run_average = 0
    change = True
    tree_order = range(settings.m_bart)

    print('Initial settings')
    print('lambda_bart value = %.3f' % param.lambda_bart)
    loglikelihood_training , mse_training = bart.compute_train_loglikelihood(data,settings,param)
    print('mse train =%.3f, loglik_train= %.3f' %(mse_training,loglikelihood_training))

    for iterator in range(settings.n_iterations):
        init_current_time = time.clock()
        if settings.verbose >= 1:
            print('\n%s BART ITERATION = %7d %s' % ('*'*30,iterator, '*'*30))

        logarithmic_prior = 0.

        if settings.sample_y == 1 and settings.mcmc_type != 'prior':
            bart.sample_labels(data,settings,param)

        bart.sample_lambda_bart(param,data,settings)
        time_sample_lambda = time.clock() -init_current_time
        logarithmic_prior += bart.lambda_logprior

        random.shuffle(tree_order)
        for ele in tree_order:
            if settings.debug == 1:
                print('\ntree_id = %3d' % ele)
            init_current_tree_time = time.clock()

            #set data['y_train'] to new value 
            bart.update_residual(ele,data)
            update_cache_tmp(cache_tmp, data , param , settings)
            
            #Get the MCMC for i_t'th tree
            bart.trees[ele].update_loglik_node_all(data,param, cache, settings)
            (bart.trees[ele],change) = run_mcmc_single_tree(bart.trees[ele],settings, data, param,cache,change,mcmc_counts,cache_tmp,bart.pmcmc_objects[ele])

            # update to new parameters
            sample_param(bart.trees[ele],settings,param)
            logarithmic_prior += bart.trees[ele].pred_val_logprior

            # update predicted value
            bart.update_predicted_value(ele,data,param,settings)

            bart.trees[ele].update_depth()
            mcmc_stats[ele,iterator,[3,6,7,8,9]] = np.array([bart.trees[ele].depth,len(bart.trees[ele].leaf_nodes),len(bart.trees[ele].non_leaf_nodes),change,time.clock()-init_current_tree_time])

            if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'grow_prune':
                mcmc_stats[ele,iterator,1] = bart.trees[ele].compute_logprior()
            else:
                mcmc_stats[ele,iterator,1] = -np.inf


        if settings.sample_y == 1 and settings.mcmc_type == 'prior':
            bart.sample_labels(data,settings,param)

        if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'growprune':
            logarithmic_prior +=float(np.sum(mcmc_stats[:,iterator,1]))
        else:
            logarithmic_prior = -np.inf
        loglikelihood_training,mse_training = bart.compute_train_loglikelihood(data,settings,param)
        bart_log_probability = logarithmic_prior + loglikelihood_training

        mcmc_stats_bart[iterator,:3]=[loglikelihood_training,logarithmic_prior,bart_log_probability]
        mcmc_stats_bart[iterator,3:7]= np.mean(mcmc_stats[:,iterator,[3,6,7,8]],0)
        mcmc_stats_bart[iterator,-3:-1]=[mse_training,param.lambda_bart]
        mcmc_stats_bart[iterator,-1] = np.sum(mcmc_stats[:,iterator,-1]) + time_sample_lambda

        if iterator == 0:
            mcmc_stats_bart[iterator,-1] += time_initialization
        if (settings.verbose >= 2):
            print('Fraction of trees where MCMC moves were accepted = %.3f' % mcmc_stats_bart[iterator,6])
        if (settings.save == 1):
            for tree_ele in bart.trees:
                tree_ele.gen_rules_tree()
            pred_tmp = {'train':bart.predict_training(data,param,settings),'test':bart.predict(data['x_test'],data['y_test_orig'],param,settings)}
            for data_of_keys in settings.perf_dataset_keys:
                for stored_keys in settings.perf_store_keys:
                    mcmc_predict_predictions[data_of_keys]['accum'][stored_keys] += pred_tmp[data_of_keys][stored_keys]
            if iterator == 0 and settings.verbose >= 1:
                print('Cumulative: itr, itr_run_avg, [mse train, logprob_train, mse test, ' 'logprob_test, time_mcmc, time_mcmc_prediction], time_mcmc_cumulative')
                print('itr, [mse train, logprob_train, mse test, ' 'logprob_test, time_mcmc, time_mcmc+time_prediction]')
            if settings.store_every_iteration == 1:
                store_every_iteration(mcmc_tree_predictions,data,settings,param,iterator,pred_tmp,mcmc_stats_bart[iterator,-1],init_current_time)
            if iterator > 0 and iterator % settings.n_run_avg == settings.n_run_avg - 1 :
                metrics={}
                for data_of_keys in settings.perf_dataset_keys:
                    k_temp,k_data_n = get_k_data_names(settings,data_of_keys)
                    for stored_keys in settings.perf_store_keys:
                        mcmc_tree_predictions[data_of_keys][stored_keys][iteration_run_average] = mcmc_tree_predictions[data_of_keys]['accum'][stored_keys]
                    metrics[data_of_keys] = compute_metrics_regression(data[k_temp],mcmc_tree_predictions[data_of_keys]['pred_mean'][iteration_run_average],mcmc_tree_predictions[data_of_keys]['pred_mean'][iteration_run_average])

                iterator_range = range(iteration_run_average*settings.n_run_avg,(iteration_run_average+1)*settings.n_run_avg)
                if settings.debug == 1:
                    print('Iteration range = %s' % iteration_range)
                mcmc_train_timing = np.sum(mcmc_stats_bart[iteration_range,-1])
                mcmc_tree_predictions['run_avg_tests'][:,iteration_run_average] = [metrics['train']['mse'],metrics['train']['log_prob'],metrics['test']['mse'],metrics['test']['log_prob'],mcmc_train_timing,time.clock()-init_time_run_average]
                if settings.verbose >= 1:
                    print('Cumulative: %7d, %7d, %s, %.2f' % (iterator,iteration_run_average,mcmc_tree_predictions['run_avg_stats'][:,iteration_run_average].T,np.sum(mcmc_tree_predictions['run_avg_stats'][-2,:iteration_run_average+1])))
                iteration_run_average += 1
                init_time_run_average = time.clock()


    print('\nTotal time in seconds =%f' % (time.clock()-init_time))
    if settings.verbose >= 2:
        print('mcmc_stats_bart[:,3:] (non cummulative) =')
        print('mean_depth,mean num_leaves ,mean num_nonleaves,mean change,mse_training,lambda_bart,time_iterations')
        print(mcmc_stats_bart[:,3:])
    if settings.verbose >=1:
        print('mean of mcmc_stats_bart discarding first 50% of the chain')
        iteration_start = mcmc_stats_bart.shape[0]/2
        for k_ele,s_ele in enumerate(mcmc_stats_bart_desc):
            print('%20s\t%.2f' %(s_ele,np.mean(mcmc_stats_bart[iteration_start:,k_ele])))
            
            
    if settings.save == 1:
        print('Averaged predictions across all previous additive trees:')
        print('mse training,mean log_prob_train, mse test,mean log_prob_test')
        print(mcmc_tree_predictions['run_avg_tests'][:4,:].T)

    if settings.save == 1:
        filename_to_use = get_filename_bart(settings)
        print('filename = '+filename_to_use)
        prediction_results={}
        prediction_results['mcmc_stats_bart'] = mcmc_stats_bart
        prediction_results['mcmc_stats_bart_desc'] = mcmc_stats_bart_desc
        if settings.store_all_stats:
            prediction_results['mcmc_stats'] = mcmc_stats
        prediction_results['settings'] = settings
        if settings.dataset[:8] == 'friedman' or settings.dataset[:3] == 'toy':
            results['data'] = data
        pickle.dump(prediction_results,open(filename_to_use,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        second_filename_to_use = filename_to_use[:-1]+ 'tree_predictions.p'
        print('predictions stored in file: %s' % second_filename_to_use)
        pickle.dump(mcmc_tree_predictions,open(second_filename_to_use,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            
            
if __name__ == "__main__":
    main()
        
                                       
        
            
            
    
