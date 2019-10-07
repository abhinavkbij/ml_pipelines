import pdb
import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer
from utils.messages_utils import append_message_flnew, append_message_adult, read_messages_count, send_retrain_message, publish_prediction_flnew, publish_prediction_adult

KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages_flnew', 'app_messages_adult', 'retrain_topic']
PATH = Path('data/')
MODELS_PATH = PATH/'models'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MESSAGES_PATH = PATH/'messages'
RETRAIN_EVERY = 25
EXTRA_MODELS_TO_KEEP = 1

column_orderFLNew = pickle.load(open(DATAPROCESSORS_PATH/'column_orderFLNew.p', 'rb'))
column_orderAdult = pickle.load(open(DATAPROCESSORS_PATH/'column_orderAdult.p', 'rb'))
dataprocessorFLNew = None
dataprocessorAdult = None
consumer = None
modelFLNew = None
modelAdult = None


def reload_model(path):
	return pickle.load(open(path, 'rb'))


def is_retraining_message(msg):
	message = json.loads(msg.value)
	return msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']


def is_application_message_flnew(msg):
	message = json.loads(msg.value)
	return msg.topic == 'app_messages_flnew' and 'prediction' not in message

def is_application_message_adult(msg):
	message = json.loads(msg.value)
	return msg.topic == 'app_messages_adult' and 'prediction' not in message


def predictFLNew(message, column_order):
	row = pd.DataFrame(message, index=[0])
	# sanity check
	assert row.columns.tolist()[:-1] == column_order
	# In the real world we would not have the target (here 'income_bracket').
	# In this example we keep it and we will retrain the model as it reads
	# RETRAIN_EVERY number of messages. In the real world, after RETRAIN_EVERY
	# number of messages have been collected, one would have to wait until we
	# can collect RETRAIN_EVERY targets AND THEN retrain
	row.drop('Proposed Policy Type', axis=1, inplace=True)
	trow = dataprocessorFLNew.transform(row)
	return modelFLNew.predict(trow)[0]

def predictAdult(message, column_order):
	row = pd.DataFrame(message, index=[0])
	# sanity check
	assert row.columns.tolist()[:-1] == column_order
	# In the real world we would not have the target (here 'income_bracket').
	# In this example we keep it and we will retrain the model as it reads
	# RETRAIN_EVERY number of messages. In the real world, after RETRAIN_EVERY
	# number of messages have been collected, one would have to wait until we
	# can collect RETRAIN_EVERY targets AND THEN retrain
	row.drop('income_bracket', axis=1, inplace=True)
	trow = dataprocessorAdult.transform(row)
	return modelAdult.predict(trow)[0]


def start(model_id, messages_count, batch_id):
	for msg in consumer:
		message = json.loads(msg.value)

		if is_retraining_message(msg):
			model_fnameFLNew = 'model_{}_FLNew.p'.format(model_id)
			modelFLNew = reload_model(MODELS_PATH/model_fnameFLNew)
			model_fnameAdult = 'model_{}_Adult.p'.format(model_id)
			modelAdult = reload_model(MODELS_PATH/model_fnameAdult)
			print("NEW MODEL RELOADED {}".format(model_id))

		elif is_application_message_flnew(msg):
			request_id = message['request_id']
			pred = predictFLNew(message['data'], column_orderFLNew)
			publish_prediction_flnew(pred, request_id)

			append_message_flnew(message['data'], MESSAGES_PATH, batch_id)
			messages_count += 1
			if messages_count % RETRAIN_EVERY == 0:
				model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
				send_retrain_message(model_id, batch_id)
				batch_id += 1

		elif is_application_message_adult(msg):
			request_id = message['request_id']
			pred = predictAdult(message['data'], column_orderAdult)
			publish_prediction_adult(pred, request_id)

			append_message_adult(message['data'], MESSAGES_PATH, batch_id)
			messages_count += 1
			if messages_count % RETRAIN_EVERY == 0:
				model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
				send_retrain_message(model_id, batch_id)
				batch_id += 1


if __name__ == '__main__':
	dataprocessor_id = 0
	dataprocessor_fnameFLNew = 'dataprocessor_{}_FLNew.p'.format(dataprocessor_id)
	dataprocessorFLNew = pickle.load(open(DATAPROCESSORS_PATH/dataprocessor_fnameFLNew, 'rb'))

	dataprocessor_fnameAdult = 'dataprocessor_{}_Adult.p'.format(dataprocessor_id)
	dataprocessorAdult = pickle.load(open(DATAPROCESSORS_PATH/dataprocessor_fnameAdult, 'rb'))

	messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
	batch_id = messages_count % RETRAIN_EVERY

	model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
	model_fnameFLNew = 'model_{}_FLNew.p'.format(model_id)
	modelFLNew = reload_model(MODELS_PATH/model_fnameFLNew)

	model_fnameAdult = 'model_{}_Adult.p'.format(model_id)
	modelAdult = reload_model(MODELS_PATH/model_fnameAdult)

	consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
	consumer.subscribe(TOPICS)

	start(model_id, messages_count, batch_id)
