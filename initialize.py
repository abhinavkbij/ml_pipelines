import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import warnings
import argparse
import os
import pdb

from pathlib import Path
from utils.preprocess_data import build_train


PATH = Path('data/')
TRAIN_PATH = PATH/'train'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'


def create_folders():
	print("creating directory structure...")
	(PATH).mkdir(exist_ok=True)
	(TRAIN_PATH).mkdir(exist_ok=True)
	(MODELS_PATH).mkdir(exist_ok=True)
	(DATAPROCESSORS_PATH).mkdir(exist_ok=True)
	(MESSAGES_PATH).mkdir(exist_ok=True)


def download_data():
	print("downloading training data...")
	df_train = pd.read_csv(PATH/'OHE_FLNew_train.csv',index_col=[0])
	df_train.to_csv(PATH/'train/train.csv',index=False)

	df_test = pd.read_csv(PATH/'OHE_FLNew_test.csv',index_col=[0])
	df_test.to_csv(PATH/'train/test.csv',index=False)


def create_data_processor():
	create_folders()
	download_data()
	print("creating preprocessor...")
	dataprocessor = build_train(TRAIN_PATH/'train.csv', DATAPROCESSORS_PATH)


def create_model(hyper):
	print("creating model...")
	init_dataprocessor = 'dataprocessor_0_.p'
	dtrain = pickle.load(open(DATAPROCESSORS_PATH/init_dataprocessor, 'rb'))
	if hyper == "hyperopt":
		# from train.train_hyperopt import LGBOptimizer
		from train.rf_hyperopt import RFOptimizer
	elif hyper == "hyperparameterhunter":
		# from train.train_hyperparameterhunter import LGBOptimizer
		return ("hyperparameterhunter not included yet!")
	RFOpt = RFOptimizer(dtrain, MODELS_PATH)
	RFOpt.optimize(maxevals=50)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--hyper", type=str, default="hyperopt")
	args = parser.parse_args()
	# create_folders()
	# download_data()
	create_data_processor()
	create_model(args.hyper)