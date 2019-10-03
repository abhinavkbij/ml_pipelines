import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pickle
import pdb
import warnings

from pathlib import Path
from sklearn.metrics import f1_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

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


class LGBOptimizer(object):
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
		print (trainDataset.data.columns)
		self.colnames = trainDataset.colnames
		self.categorical_columns = trainDataset.categorical_columns
		print (trainDataset.categorical_columns)

		self.lgtrain = lgb.Dataset(self.X,label=self.y,
			feature_name=self.colnames,
			categorical_feature = self.categorical_columns,
			free_raw_data=False)

	def optimize(self, maxevals=200, model_id=0):

		param_space = self.hyperparameter_space()
		objective = self.get_objective(self.lgtrain)
		objective.i=0
		trials = Trials()
		print ("-----========")
		best = fmin(fn=objective,
		            space=param_space,
		            algo=tpe.suggest,
		            max_evals=maxevals,
		            trials=trials)
		best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
		best['num_leaves'] = int(best['num_leaves'])
		best['verbose'] = -1
		print ("=====-------")
		# set the model with the best parameters, fit and save
		model = lgb.LGBMClassifier(**best)
		model.fit(self.lgtrain.data,
			self.lgtrain.label,
			feature_name=self.colnames,
			categorical_feature=self.categorical_columns)

		model_fname = 'model_{}_.p'.format(model_id)
		best_experiment_fname = 'best_experiment_{}_.p'.format(model_id)

		pickle.dump(model, open(self.PATH/model_fname, 'wb'))
		pickle.dump(best, open(self.PATH/best_experiment_fname, 'wb'))

		self.best = best
		self.model = model


	def get_objective(self, train):

		def objective(params):
			"""
			objective function for lightgbm.
			"""
			# hyperopt casts as float
			params['num_boost_round'] = int(params['num_boost_round'])
			params['num_leaves'] = int(params['num_leaves'])

			# need to be passed as parameter
			params['objective'] = 'multiclass'
			params['is_unbalance'] = True
			params['verbose'] = -1
			params['seed'] = 1
			params['num_classes'] = 8
			print ("-----sdjflfsdlsfl-----")
			cv_result = lgb.cv(
				params,
				train,
				num_boost_round=params['num_boost_round'],
				metrics='multi_logloss',
				# feval = lgb_f1_score,
				nfold=3,
				stratified=True,
				early_stopping_rounds=20)
			print (cv_result)
			self.early_stop_dict[objective.i] = len(cv_result['multi_logloss-mean'])
			error = round(cv_result['multi_logloss-mean'][-1], 4)
			objective.i+=1
			print ("-----------------")
			return error

		return objective

	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
			'num_boost_round': hp.quniform('num_boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
		    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
		}

		if param_space:
			return param_space
		else:
			return space






# class RFOptimizer(object):
# 	def __init__(self, trainDataset, out_dir):
# 		self.PATH = out_dir
# 		self.X = trainDataset.data
# 		self.y = trainDataset.target
# 		self.best = 0
# 		self.space4rf = {
# 		'max_depth': hp.choice('max_depth', range(1,20)),
# 		'max_features': hp.choice('max_features', range(1,5)),
# 		'n_estimators': hp.choice('n_estimators', range(1,20)),
# 		'criterion': hp.choice('criterion', ["gini", "entropy"]),
# 		# 'scale': hp.choice('scale', [0, 1]),
# 		# 'normalize': hp.choice('normalize', [0, 1])
# 		}
	
# 	def optimize(self,model_id=0):
# 		def hyperopt_train_test(params):
# 			X_ = self.X[:]
# 			if 'normalize' in params:
# 				if params['normalize'] == 1:
# 					X_ = normalize(X_)
# 					del params['normalize']
# 			if 'scale' in params:
# 				if params['scale'] == 1:
# 					X_ = scale(X_)
# 					del params['scale']
# 			clf = RandomForestClassifier(**params)
# 			return cross_val_score(clf, self.X, self.y).mean()
# 		def f(self,params):
# 			acc = hyperopt_train_test(params)
# 			if acc > self.best:
# 				self.best = acc
# 			print ('new best:', self.best, params)
# 			return {'loss': -acc, 'status': STATUS_OK}
# 		trials = Trials()
# 		self.best = fmin(f, self.space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
# 		model = RandomForestClassifier(**self.best)
# 		model.fit(self.X,
# 				self.y)

# 		model_fname = 'model_{}_.p'.format(model_id)
# 		best_experiment_fname = 'best_experiment_{}_.p'.format(model_id)

# 		pickle.dump(model, open(self.PATH/model_fname, 'wb'))
# 		pickle.dump(best, open(self.PATH/best_experiment_fname, 'wb'))

# 		self.best = best
# 		self.model = model

