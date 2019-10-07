import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import pickle
import pdb
import warnings

from pathlib import Path
from sklearn.metrics import f1_score
from hyperopt import hp, tpe, fmin, Trials
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def best_threshold(y_true, pred_proba, proba_range, verbose=False):
	"""
	Function to find the probability threshold that optimises the f1_score

	Comment: this function is not used in this excercise, but we include it in
	case the reader finds it useful

	Parameters:
	-----------
	y_true: numpy.ndarray
		array with the true labels
	pred_proba: numpy.ndarray
		array with the predicted probability
	proba_range: numpy.ndarray
		range of probabilities to explore.
		e.g. np.arange(0.1,0.9,0.01)

	Return:
	-----------
	tuple with the optimal threshold and the corresponding f1_score
	"""
	scores = []
	for prob in proba_range:
		pred = [int(p>prob) for p in pred_proba]
		score = f1_score(y_true,pred)
		scores.append(score)
		if verbose:
			print("INFO: prob threshold: {}.  score :{}".format(round(prob,3), round(score,5)))
	best_score = scores[np.argmax(scores)]
	optimal_threshold = proba_range[np.argmax(scores)]
	return (optimal_threshold, best_score)


def lgb_f1_score(preds, lgbDataset):
	"""
	Function to compute the f1_score to be used with lightgbm methods.
	Comments: output format must be:
	(eval_name, eval_result, is_higher_better)

	Parameters:
	-----------
	preds: np.array or List
	lgbDataset: lightgbm.Dataset
	"""
	binary_preds = [int(p>0.5) for p in preds]
	y_true = lgbDataset.get_label()
	# lightgbm: (eval_name, eval_result, is_higher_better)
	return 'f1', f1_score(y_true, binary_preds), True


class RFOptimizer(object):
	def __init__(self, trainDataset, out_dir):
		"""
		Hyper Parameter optimization

		Parameters:
		-----------
		trainDataset: FeatureTools object
			The result of running FeatureTools().fit()
		out_dir: pathlib.PosixPath
			Path to the output directory
		"""
		self.PATH = out_dir
		self.early_stop_dict = {}

		self.X = trainDataset.data
		self.y = trainDataset.target
		self.colnames = trainDataset.colnames
		self.best = 0
		

	def optimize(self, maxevals=200, model_id=0):

		param_space = self.hyperparameter_space()
		objective = self.get_objective()
		objective.i=0
		trials = Trials()
		best = fmin(fn=objective,
					space=param_space,
					algo=tpe.suggest,
					max_evals=maxevals,
					trials=trials)
		# best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
		# best['num_leaves'] = int(best['num_leaves'])
		# best['verbose'] = -1
		best['criterion'] = ['gini','entropy'][best['criterion']]
		best['n_jobs'] = 3

		# set the model with the best parameters, fit and save
		model = RandomForestClassifier(**best)
		model.fit(self.X, self.y)

		model_fname = 'model_{}_FLNew.p'.format(model_id)
		best_experiment_fname = 'best_experiment_{}_FLNew.p'.format(model_id)

		pickle.dump(model, open(self.PATH/model_fname, 'wb'))
		pickle.dump(best, open(self.PATH/best_experiment_fname, 'wb'))

		self.best = best
		self.model = model


	def get_objective(self):

		def objective(params):
			"""
			objective function for lightgbm.
			"""
			clf = RandomForestClassifier(**params)
			acc = cross_val_score(clf, self.X, self.y).mean()
			if acc > self.best:
				self.best = acc
			# print ('new best:', self.best, params)
			error = 1-acc
			return error

		return objective

	def hyperparameter_space(self, param_space=None):

		space = {
			'max_depth': hp.choice('max_depth', range(1,20)),
			'max_features': hp.choice('max_features', range(1,419)),
			'n_estimators': hp.choice('n_estimators', range(1,5000)),
			'criterion': hp.choice('criterion', ["gini", "entropy"])
		}

		if param_space:
			return param_space
		else:
			return space