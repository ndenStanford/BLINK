import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import blink.main_dense as main_dense
from blink.biencoder.biencoder import *
import argparse
import torch
import numpy as np 
import json 

import time
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import * 


models_path = "model_support/"#"/home/ubuntu/BLINK/models/" # the path where you stored the BLINK models

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

data_to_link = generate_complex_query()

models = main_dense.load_models(args, logger=None)

_, _, _, _, _, predictions, scores, mention_found = main_dense.run(args, None, *models, test_data=data_to_link, REL_filter=True)

entity_linking_dict = dict()
for (m, p, d) in zip(mention_found, predictions, data_to_link):
    if m:
        entity_linking_dict[d['mention']] = p[0]

print(entity_linking_dict)




"""
(
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
) = main_dense._load_candidates(
    args.entity_catalogue, 
    args.entity_encoding, 
    faiss_index=getattr(args, 'faiss_index', None), 
    index_path=getattr(args, 'index_path' , None),
    logger=None,
)


with open(args.biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = args.biencoder_model
#biencoder = load_biencoder(biencoder_params)
biencoder = load_biencoder_inference(biencoder_params)

cand_encs = torch.load('sample_input/cand_encs.pt')
text_vecs = torch.load('sample_input/text_vecs.pt')

model = biencoder
inputs = [text_vecs, cand_encs]

x=model(*inputs)
#biencoder = torch.jit.load('biencoder.pt')

crossencoder = None 
crossencoder_params = None 

models = biencoder, biencoder_params, crossencoder, crossencoder_params, candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id, faiss_indexer
"""