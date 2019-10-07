import pandas as pd
import json
import threading
import uuid

from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from time import sleep


PATH = Path('data/')
KAFKA_HOST = 'localhost:9092'
cols = ['ProductName','Country','LocalLeaderShip','LocalEmployees','BuyerProfession','ProofOfCoverage','LocalServers','NoOfClaims','Proposed Policy Type']
df_testFLNew = pd.read_csv(PATH/'train/testFLNew.csv')
df_testAdult = pd.read_csv(PATH/'adult.test')
# In the real world, the messages would not come with the target/outcome of
# our actions. Here we will keep it and assume that at some point in the
# future we can collect the outcome and monitor how our algorithm is doing
# df_test.drop('income_bracket', axis=1, inplace=True)
df_testFLNew['json'] = df_testFLNew.apply(lambda x: x.to_json(), axis=1)
df_testAdult['json'] = df_testAdult.apply(lambda x: x.to_json(), axis=1)

messagesFLNew = df_testFLNew.json.tolist()
messagesAdult = df_testAdult.json.tolist()


def start_producing():
	producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
	for i in range(200):
		message_id = str(uuid.uuid4())
		messageFLNew = {'request_id': message_id, 'data': json.loads(messagesFLNew[i])}
		messageAdult = {'request_id': message_id, 'data': json.loads(messagesAdult[i])}


		producer.send('app_messages_flnew', json.dumps(messageFLNew).encode('utf-8'))
		producer.flush()

		producer.send('app_messages_adult', json.dumps(messageAdult).encode('utf-8'))
		producer.flush()

		print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
		sleep(2)


def start_consuming_flnew():
	consumer = KafkaConsumer('app_messages_flnew', bootstrap_servers=KAFKA_HOST)
	print ("flnew consumer: ")
	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			request_id = message['request_id']
			print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))

def start_consuming_adult():
	consumer = KafkaConsumer('app_messages_adult', bootstrap_servers=KAFKA_HOST)
	print ("adult consumer: ")
	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			request_id = message['request_id']
			print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))


threads = []
t = threading.Thread(target=start_producing)
t1 = threading.Thread(target=start_consuming_flnew)
t2 = threading.Thread(target=start_consuming_adult)
threads.append(t)
threads.append(t1)
threads.append(t2)
t.start()
t1.start()
t2.start()
