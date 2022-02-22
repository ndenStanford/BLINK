import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import numpy as np
import os
import torch_neuron
import blink.main_dense as main_dense
import argparse
import json 
import time

print(torch_neuron.__version__)

cand_encs = torch.load('sample_input/cand_encs_cpu.pt')
text_vecs = torch.load('sample_input/text_vecs_cpu.pt')
inputs = [text_vecs, cand_encs]

module_loaded_neuron=torch.jit.load("blink_entity_linking_neuron.pt")
x=module_loaded_neuron.forward(*inputs)
print(x)


models_path = "/home/ubuntu/BLINK/torchserve_sandbox/model_support"#'/opt/models/model-support/'#"model_store/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"/biencoder_wiki_large.bin",
    "biencoder_config": models_path+"/biencoder_wiki_large.json",
    "entity_catalogue": models_path+"/entity.jsonl",
    "entity_encoding": models_path+"/all_entities_large.t7",
    "crossencoder_model": models_path+"/crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"/crossencoder_wiki_large.json",
    "fast": True, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

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

models = main_dense.load_models(args, logger=None)

biencoder, biencoder_params, crossencoder, crossencoder_params, candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id, faiss_indexer = models

models_jit_neuron = module_loaded_neuron, biencoder_params, crossencoder, crossencoder_params, candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id, faiss_indexer 

_, _, _, _, _, predictions, scores, mention_found = main_dense.run(args, None, *models_jit_neuron, test_data=data_to_link, REL_filter=True)
#_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link, REL_filter=True)
print(predictions)