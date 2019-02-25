import pandas as pd
import json

from pathlib import Path
from kafka import KafkaProducer
from time import sleep

PATH = Path('data/')
df_test = pd.read_csv(PATH/'adult.test')
df_test.drop('income_bracket', axis=1, inplace=True)
df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)
messages = df_test.json.tolist()

producer = KafkaProducer(bootstrap_servers='localhost:9092')
for i in range(10):
    producer.send('my_favorite_topic', messages[i].encode('utf-8'))
    producer.flush()
    # sleep(0.5)