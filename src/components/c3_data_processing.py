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
        self.vocab = []
        self.vocab_size = 0
        logging.info("DataProcessing class Initialization completed")
    
    def tokenize(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()
    
    def Process(self):
        logging.info("Entered Method Process")
        df = pd.read_csv(self.config.X_path)
        df['tok_1'] = df['question1'].apply(self.tokenize)
        df['tok_2'] = df['question2'].apply(self.tokenize)
        
        all_sentences = df['tok_1'].tolist() + df['tok_2'].tolist()
        counter = Counter([word for sent in all_sentences for word in sent])
        self.vocab = ['<pad>', '<unk>'] + [word for word, count in counter.items() if count >= 5]
        self.vocab_size = len(self.vocab)
        
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        enc_sentences = [[self.word2idx.get(word, 1) for word in sent] for sent in all_sentences]
        
        logging.info("Exited Method Process")


'''
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

    def save_encoded_questions(self, df):
        logging.info("Entered Method save_encoded_questions")
        q1_encoded_path = self.config.q1_encoded
        q2_encoded_path = self.config.q2_encoded
        np.savetxt(q1_encoded_path, df['q1_encoded'].values, delimiter=",", fmt="%d")
        np.savetxt(q2_encoded_path, df['q2_encoded'].values, delimiter=", ", fmt="%d")
        logging.info("Exited Method save_encoded_questions")
        
    def build_skipgram_pairs(self):
        logging.info("Entered Method build_skipgram_pairs")
        window_size = self.config.window_size
        
        logging.info("Entered Method build_skipgram_pairs")
'''