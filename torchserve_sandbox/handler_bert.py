import os
import json
import sys
import logging

import torch
from transformers import AutoTokenizer
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler

import blink.main_dense as main_dense
import argparse
import torch
import numpy as np 

import time


# one core per worker
# os.environ['NEURONCORE_GROUP_SIZES'] = '1'

logger = logging.getLogger(__name__)

class BertEmbeddingHandler(BaseHandler, ABC):
    """
    Handler class for Bert Embedding computations.
    """
    def __init__(self):
        super(BertEmbeddingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        print('starting initializing')
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = 'cpu'
        model_dir = properties.get('model_dir')
        print('model_dir: ', model_dir)
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        # point sys.path to our config file
        with open('config.json') as fp:
            config = json.load(fp)
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        #self.classes = ['not paraphrase', 'paraphrase']

        print('starting loading model')
        biencoder = torch.jit.load(model_pt_path)
        logger.debug(f'Model loaded from {model_dir}')
        print('model loaded')
        #self.model.to(self.device)
        #self.model.eval()
        
        #models_path = "/home/ubuntu/BLINK/models/" # the path where you stored the BLINK models
        print('before loading support materials')
        model_support_path = "/home/ubuntu/BLINK/torchserve_sandbox/model_support"#model_dir#'/opt/models/model-support'#

        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 10,
            "biencoder_model": model_support_path+"/biencoder_wiki_large.bin",
            "biencoder_config": model_support_path+"/biencoder_wiki_large.json",
            "entity_catalogue": model_support_path+"/entity.jsonl",
            "entity_encoding": model_support_path+"/all_entities_large.t7",
            "crossencoder_model": model_support_path+"/crossencoder_wiki_large.bin",
            "crossencoder_config": model_support_path+"/crossencoder_wiki_large.json",
            "fast": True, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
        }

        self.args = argparse.Namespace(**config)

        (
            candidate_encoding,
            title2id,
            id2title,
            id2text,
            wikipedia_id2local_id,
            faiss_indexer,
        ) = main_dense._load_candidates(
            self.args.entity_catalogue, 
            self.args.entity_encoding, 
            faiss_index=getattr(self.args, 'faiss_index', None), 
            index_path=getattr(self.args, 'index_path' , None),
            logger=None,
        )

        with open(self.args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
            biencoder_params["path_to_model"] = self.args.biencoder_model

        crossencoder = None 
        crossencoder_params = None 

        self.models = biencoder, biencoder_params, crossencoder, crossencoder_params, candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id, faiss_indexer
        self.initialized = True

        print('end initializing')


    def preprocess(self, input_data):
        """
        Tokenization pre-processing
        """

        return input_data

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        data_to_link=inputs[0]['body']

        _, _, _, _, _, predictions, scores, mention_found = main_dense.run(self.args, None, *self.models, test_data=data_to_link, REL_filter=True)
        
        entity_linking_dict = dict()
        for (m, p, d) in zip(mention_found, predictions, data_to_link):
            if m:
                entity_linking_dict[d['mention']] = p[0]
        
        return [entity_linking_dict] #[predictions]

    def postprocess(self, inference_output):
        return inference_output

