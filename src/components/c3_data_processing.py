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
import tqdm
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkipGramCustomDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

class SkipGramEmbModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.uniform_(self.out_embed.weight, -0.5/emb_dim, 0.5/emb_dim)

    def forward(self, center, context, neg_samples):
        v_center = self.in_embed(center)
        v_out = self.out_embed(context)
        v_neg = self.out_embed(neg_samples)
        pos_score = torch.sum(v_center * v_out, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        neg_score = torch.bmm(v_neg, v_center.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)
        return -(pos_loss + neg_loss).mean()
        

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
    
    def generate_pairs(self, enc_sentences, window_size=3):
        pairs = []
        for enc_sent in enc_sentences:
            sent_len = len(enc_sent)
            for idx, cen_word in enumerate(enc_sent):
                lo = max(0, idx - window_size)
                hi = min(sent_len, idx + window_size + 1)
                for j in range(lo, hi):
                    if j != idx:
                        pairs.append((cen_word, enc_sent[j]))
        return pairs
    
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
        pairs = self.generate_pairs(enc_sentences, window_size=self.config.emb_window_size)
        
        word_freq = np.array([counter[self.idx2word[i]] for i in range(self.vocab_size)])
        neg_sampling = word_freq ** (3/4)
        neg_sampling /= neg_sampling.sum()
        neg_sampling_tensor = torch.tensor(neg_sampling, dtype=torch.float)
        
        #create embedding model
        model = SkipGramEmbModel(self.vocab_size, self.config.emb_dim).to(DEVICE)
        torch.save(model, self.config.base_emb_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dataset = SkipGramCustomDataset(pairs)
        loader = DataLoader(dataset, batch_size=self.config.emb_batch_size, shuffle=True)
        neg_sampling_tensor = torch.tensor(neg_sampling, dtype=torch.float)
        
        #run model
        for epoch in range(self.config.emb_epochs):
            total_loss = 0
            for center, context in tqdm(loader, desc="Generating Embeddings..."):
                center = center.to(DEVICE)
                context = context.to(DEVICE)
                neg_samples = torch.multinomial(neg_sampling_tensor, center.size(0) * self.config.emb_neg_samples, replacement=True)
                neg_samples = neg_samples.view(center.size(0), self.config.emb_neg_samples).to(DEVICE)

                loss = model(center, context, neg_samples)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        torch.save(model, self.config.trained_emb_model)
        
        embedding_matrix = model.in_embed.weight.detach()
        embedding_np = embedding_matrix.cpu().numpy()
        embedding_df = pd.DataFrame(embedding_np,
                                columns=[f"dim_{i}" for i in range(embedding_np.shape[1])])
        embedding_df.insert(0, 'word', self.vocab)
        embedding_df.to_csv('word_pair_emb_matrix.csv', index=False)
        
        logging.info("Exited Method Process")