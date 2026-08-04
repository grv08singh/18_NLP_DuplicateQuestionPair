import cudf
import pandas as pd
import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from src.config.configuration import DataProcessingConfig
from src.logger import logging

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        logging.info("DataProcessing class Initialization started")
        self.config = config
        self.corpus = []
        self.corpus_size = 0
        self.counter = Counter()
        self.vocab_list = []
        self.vocab_size = 0
        self.vocab_with_idx = {}
        self.token_list = []
        self.pairs = []
        logging.info("DataProcessing class Initialization completed")
        
    def save_y(self, df):
        logging.info("Entered Method save_y")
        df['is_duplicate'].to_csv(self.config.y_path, index=False)
        logging.info("Exited Method save_y")
    
    def split_txt(self, text):
        logging.info("Entered Method split_txt")
        logging.info("Exited Method split_txt")
        return text.split()
    
    def build_corpus(self, df):
        logging.info("Entered Method build_corpus")
        for col in ['question1', 'question2']:
            for row in df[col]:
                row_tokens_list = self.split_txt(row)
                self.corpus.extend(row_tokens_list)
        self.corpus_size = len(self.corpus)
        logging.info("Exited Method build_corpus")
    
    def token_counter(self):
        #counts how many times a unique token appears in total
        logging.info("Entered Method token_counter")
        self.counter = Counter(self.corpus)
        logging.info("Exited Method token_counter")
    
    def log_token_counts(self):
        #check how many times each token appears
        logging.info("Entered Method print_token_counts")
        for token, count in self.counter.items():
            logging.info(f"Token: {token}, Count: {count}")
        logging.info("Exited Method print_token_counts")
    
    def build_vocab_list(self):
        logging.info("Entered Method build_vocab_list")
        #builds vocabulary list from token counter object
        self.vocab_list = ['<pad>','<unk>'] + [token for token, count in self.counter.items() if count >= self.config.min_token_count]
        self.vocab_size = len(self.vocab_list)
        logging.info("Exited Method build_vocab_list")
        
    def build_vocab_idx(self):
        #builds a dictionary with vocabulary index
        logging.info("Entered Method build_vocab_idx")
        self.vocab_with_idx = {w:i for i,w in enumerate(self.vocab_list)}
        logging.info("Exited Method build_vocab_idx")
    
    def encode_sentence(self, sent):
        #encodes a sentence into a list of vocabulary indices
        logging.info("Entered Method encode_sentence")
        max_words = self.config.max_words
        vocab = self.vocab_with_idx    #dictionary of token:idx
        unk_word_idx = vocab['<unk>']
        encoded_sent = [vocab.get(token, unk_word_idx) for token in sent]
        encoded_sent = encoded_sent[:max_words]    #cap the no. of words
        encoded_sent += [0] * (max_words - len(encoded_sent))    #pad with zeros to make sentence of length 50 tokens
        logging.info("Exited Method encode_sentence")
        return encoded_sent

    def encode_df(self, df):
        logging.info("Entered Method encode_df")
        df['q1_encoded'] = df['question1'].apply(lambda s: self.encode_sentence(s))
        df['q2_encoded'] = df['question2'].apply(lambda s: self.encode_sentence(s))
        logging.info("Exited Method encode_df")
        return df

    def save_encoded_questions(self, df):
        logging.info("Entered Method save_encoded_questions")
        q1_encoded_path = self.config.q1_encoded
        q2_encoded_path = self.config.q2_encoded
        np.savetxt(q1_encoded_path, df['q1_encoded'].values, delimiter=",", fmt="%d")
        np.savetxt(q2_encoded_path, df['q2_encoded'].values, delimiter=", ", fmt="%d")
        logging.info("Exited Method save_encoded_questions")
        
    def build_skipgram_pairs(self):
        logging.info("Entered Method build_skipgram_pairs")
        window = self.config.token_window
        
        logging.info("Entered Method build_skipgram_pairs")