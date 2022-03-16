import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import json
import concurrent.futures
import requests
import time

from nltk.tokenize import sent_tokenize, word_tokenize
from elasticsearch import Elasticsearch
from utils import * 

with open('config.json') as fp:
    config = json.load(fp)
max_length = config['max_length']
batch_size = 1#config['batch_size']
name = 'blink_entity_linking'
url = f'http://localhost:8080/predictions/{name}'


es = Elasticsearch(['https://crawler-prod:GnVjrB5jXgGGzPZHWNRpwWGu4NqTWJsw@search5-client.airpr.com'], verify_certs=False)
body = {
          "query" :{
              "term": {
                        "lang": "en"
                }
          },
          "size": 500
        }

result = es.search(index='crawler-2021.12', body=body)

for r in result['hits']['hits']:
    text = r['_source']['content']
    
    sentences = sent_tokenize(text)

    for i in range(len(sentences)):
        sentences[i] = text_preprocessor(sentences[i], 'en')

    entities = r['_source']['entities']
    data = generate_sample_es(entities, sentences)

    if len(data) > 0:
        start = time.time()
        response = requests.post(url, json=data)
        predictions = response.json()
        print(predictions)
        end = time.time()
        print(end - start)




"""
with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    def worker_thread(worker_index):
        # we'll send half the requests as not_paraphrase examples for sanity
        data = paraphrase if worker_index < batch_size//2 else not_paraphrase
        response = requests.post(url, data=data)
        print(worker_index, response.json())

    for worker_index in range(batch_size):
        executor.submit(worker_thread, worker_index)
"""