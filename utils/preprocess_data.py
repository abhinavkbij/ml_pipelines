import pandas as pd
import pickle
import json
import pdb
import warnings

from pathlib import Path
from utils.feature_tools import FeatureTools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def load_new_training_data(path):
	data = []
	with open(path, "r") as f:
		for line in f:
			data.append(json.loads(line))
	return pd.DataFrame(data)

def build_train(train_path, results_path, dataprocessor_id=0, PATH_2=None):
	target = 'Proposed Policy Type'
	# read initial DataFrame
	cols = ['ProductName','Country','LocalLeaderShip','LocalEmployees','BuyerProfession','ProofOfCoverage','LocalServers','NoOfClaims','Proposed Policy Type']
	df = pd.read_csv(train_path)
	if PATH_2:
		df_tmp = load_new_training_data(PATH_2)
		# Let's make sure columns are in the same order
		df_tmp = df_tmp[df.columns]
		# append new DataFrame
		df = pd.concat([df, df_tmp], ignore_index=True)
		# Save it to disk
		df.to_csv(train_path,index=False)

	# categorical_columns = list(df.drop(target,axis=1).select_dtypes(include=['object']).columns)
	# numerical_columns = [c for c in df.columns if c not in categorical_columns+[target]]
	# crossed_columns = (['education', 'occupation'], ['native_country', 'occupation'])

	preprocessor = FeatureTools()
	dataprocessor = preprocessor.fit(
		df,
		target,
		)

	dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
	pickle.dump(dataprocessor, open(results_path/dataprocessor_fname, "wb"))
	if dataprocessor_id==0:
		pickle.dump(df.columns.tolist()[:-1], open(results_path/'column_order.p', "wb"))

	return dataprocessor


# if __name__ == '__main__':

# 	PATH = Path('data/')
# 	TRAIN_PATH = PATH/'train'
# 	DATAPROCESSORS_PATH = PATH/'dataprocessors'

# 	dataprocessor = build_train(TRAIN_PATH/'train.csv', DATAPROCESSORS_PATH)

