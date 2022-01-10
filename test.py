import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import blink.main_dense as main_dense
import argparse
import torch
import numpy as np 

import time
from utils import * 


models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": True, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

#biencoder, biencoder_params, crossencoder, crossencoder_params, candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id, faiss_indexer = models

"""
data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                }
                ]
"""

text = """Tesla's stock (TSLA) has a clear shot to more fertile grounds, contends long-time bullish analyst Dan Ives at Wedbush. 

"Demand for China is the linchpin. As capacity builds in Berlin and Austin, that's what I think sends Tesla's stock to $1,400 as our base case. Our bull case is $1,800," Ives said on Yahoo Finance Live. Ives rates Tesla at Outperform, and is one of the most upbeat analysts on the Street on the EV maker. 

Tesla shares traded at $1,080 as of this writing. 

Ives' hearty price target on Tesla's stock is a function of two factors.

First, Ives estimates 40% of Tesla's deliveries in 2022 will be derived from the lucrative China market. And two, the supply chain issues (namely semiconductor shortages) that have plagued automakers this year will abate in 2022. In turn, Tesla stands to surprise the Street by delivering close to 1.5 million units by year-end.

The return to a focus on Tesla's fundamentals would be welcome news for the automaker's bulls.

Tesla shares have come under pressure in December as CEO Elon Musk sells down his stake in the company to meet tax obligations. Musk has sold roughly 15.6 million shares for a shade over $16 billion, bringing him close to unloading 10% of his stake in the company as planned."""

sentences = sent_tokenize(text)

entities = {'entities': [{'entity_tokens': ['TS', '##LA'], 'entity_type': 'org', 'entity_text': 'TSLA', 'sentence_indexes': [0]}, {'entity_tokens': ['Dan', 'I', '##ves'], 'entity_type': 'per', 'entity_text': 'Dan Ives', 'sentence_indexes': [0]}, {'entity_tokens': ['We', '##dbu', '##sh'], 'entity_type': 'org', 'entity_text': 'Wedbush', 'sentence_indexes': [0]}, {'entity_tokens': ['China'], 'entity_type': 'geo', 'entity_text': 'China', 'sentence_indexes': [1, 7]}, {'entity_tokens': ['Berlin'], 'entity_type': 'geo', 'entity_text': 'Berlin', 'sentence_indexes': [2]}, {'entity_tokens': ['Austin'], 'entity_type': 'geo', 'entity_text': 'Austin', 'sentence_indexes': [2]}, {'entity_tokens': ['Tesla', '##s'], 'entity_type': 'per', 'entity_text': 'Teslas', 'sentence_indexes': [0, 2, 6, 7, 10]}, {'entity_tokens': ['1800'], 'entity_type': 'tim', 'entity_text': '1800', 'sentence_indexes': [3]}, {'entity_tokens': ['I', '##ves'], 'entity_type': 'per', 'entity_text': 'Ives', 'sentence_indexes': [0, 3, 4, 6, 7]}, {'entity_tokens': ['Yahoo', 'Finance', 'Live'], 'entity_type': 'org', 'entity_text': 'Yahoo Finance Live', 'sentence_indexes': [3]}, {'entity_tokens': ['Tesla'], 'entity_type': 'per', 'entity_text': 'Tesla', 'sentence_indexes': [0, 2, 4, 5, 6, 7, 9, 10, 11]}, {'entity_tokens': ['Out', '##per', '##form'], 'entity_type': 'org', 'entity_text': 'Outperform', 'sentence_indexes': [4]}, {'entity_tokens': ['EV'], 'entity_type': 'org', 'entity_text': 'EV', 'sentence_indexes': [4]}, {'entity_tokens': ['Tesla'], 'entity_type': 'org', 'entity_text': 'Tesla', 'sentence_indexes': [0, 2, 4, 5, 6, 7, 9, 10, 11]}, {'entity_tokens': ['Tesla', '##s'], 'entity_type': 'per', 'entity_text': 'Teslas', 'sentence_indexes': [0, 2, 6, 7, 10]}, {'entity_tokens': ['I', '##ves'], 'entity_type': 'gpe', 'entity_text': 'Ives', 'sentence_indexes': [0, 3, 4, 6, 7]}, {'entity_tokens': ['Tesla', '##s'], 'entity_type': 'org', 'entity_text': 'Teslas', 'sentence_indexes': [0, 2, 6, 7, 10]}, {'entity_tokens': ['2022'], 'entity_type': 'tim', 'entity_text': '2022', 'sentence_indexes': [7, 8]}, {'entity_tokens': ['China'], 'entity_type': 'geo', 'entity_text': 'China', 'sentence_indexes': [1, 7]}, {'entity_tokens': ['2022'], 'entity_type': 'tim', 'entity_text': '2022', 'sentence_indexes': [7, 8]}, {'entity_tokens': ['Tesla'], 'entity_type': 'per', 'entity_text': 'Tesla', 'sentence_indexes': [0, 2, 4, 5, 6, 7, 9, 10, 11]}, {'entity_tokens': ['Street'], 'entity_type': 'tim', 'entity_text': 'Street', 'sentence_indexes': [4, 9]}, {'entity_tokens': ['Tesla', '##s'], 'entity_type': 'per', 'entity_text': 'Teslas', 'sentence_indexes': [0, 2, 6, 7, 10]}, {'entity_tokens': ['Tesla'], 'entity_type': 'org', 'entity_text': 'Tesla', 'sentence_indexes': [0, 2, 4, 5, 6, 7, 9, 10, 11]}, {'entity_tokens': ['December'], 'entity_type': 'tim', 'entity_text': 'December', 'sentence_indexes': [11]}, {'entity_tokens': ['Elo', '##n', 'Mus', '##k'], 'entity_type': 'per', 'entity_text': 'Elon Musk', 'sentence_indexes': [11]}, {'entity_tokens': ['Mus', '##k'], 'entity_type': 'per', 'entity_text': 'Musk', 'sentence_indexes': [11, 12]}]}

entities = update_entities(entities, sentences)
_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link, REL_filter=True)
print(predictions)
