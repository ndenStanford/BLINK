import os
import json
import sys
import logging

import torch, torch_neuron
from transformers import AutoTokenizer
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler

import blink.main_dense as main_dense
import argparse
import torch
import numpy as np 

import time
from nltk.tokenize import sent_tokenize, word_tokenize

# one core per worker
os.environ['NEURONCORE_GROUP_SIZES'] = '1'

logger = logging.getLogger(__name__)

class BertEmbeddingHandler(BaseHandler, ABC):
    """
    Handler class for Bert Embedding computations.
    """
    def __init__(self):
        super(BertEmbeddingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        models_path = "../models/" # the path where you stored the BLINK models

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

        self.args = argparse.Namespace(**config)

        self.models = main_dense.load_models(args, logger=None)
        self.initialized = True

    def preprocess(self, input_data):
        """
        Tokenization pre-processing
        """

        return input_data

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """

        _, _, _, _, _, predictions, scores, = main_dense.run(self.args, None, *self.models, test_data=inputs, REL_filter=True)
        
        return predictions

    def postprocess(self, inference_output):
        return inference_output

