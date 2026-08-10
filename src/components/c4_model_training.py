from src.logger import logging
from src.entities.config_entity import TrainingConfig
from src.utils.common import save_json, load_json
#%load_ext cudf.pandas
#%load_ext cuml.accel

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, q1, q2, y):
        self.q1 = torch.tensor(q1, dtype=torch.float32)
        self.q2 = torch.tensor(q2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.q1[idx], self.q2[idx], self.y[idx]
    
class SiameseGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        combined_dim = hidden_dim * 2 * 3 + 1  # h1 + h2 + abs_diff + cos_sim scalar
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def encode(self, X):
        _, h = self.gru(X)
        return torch.cat([h[0], h[1]], dim=1)
    
    def forward(self, q1, q2):
        h1, h2 = self.encode(q1), self.encode(q2)
        cos_sim = F.cosine_similarity(h1, h2).unsqueeze(1)
        abs_diff = torch.abs(h1 - h2)
        comb_tensor = torch.cat([h1, h2, cos_sim, abs_diff], dim=1)
        return self.classifier(comb_tensor).squeeze(1)

class Training:
    def __init__(self, config: TrainingConfig):
        logging.info("PrepareBaseModel class Initialization started")
        self.config = config
        logging.info("PrepareBaseModel class Initialization completed")
    
    def reshape_to_seq(self, emb_matrix, seq_len):
        logging.info("Entered Method reshape_to_seq")
        # reshaping to match gru input args
        dim = emb_matrix.shape[1]
        assert dim % seq_len == 0
        logging.info("Exited Method reshape_to_seq")
        return emb_matrix.reshape(emb_matrix.shape[0], seq_len, dim // seq_len)
    
    def train_gru(self, emb_q1, emb_q2, y, trained_model_path, val_rep_path, seq_len=8):
        logging.info("Entered Method train_gru")
        epochs = self.config.epochs
        X1 = self.reshape_to_seq(emb_q1, seq_len)
        X2 = self.reshape_to_seq(emb_q2, seq_len)
        
        X1_tr, X1_val, X2_tr, X2_val, y_tr, y_val = train_test_split(
            X1, X2, y, test_size=0.15, random_state=42, stratify=y)

        tr_dataset = CustomDataset(X1_tr, X2_tr, y_tr)
        val_dataset = CustomDataset(X1_val, X2_val, y_val)
        
        tr_loader = DataLoader(tr_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = SiameseGRU(input_dim=X1.shape[2]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        def run_epoch(loader, train=True):
            model.train() if train else model.eval()
            total_loss, correct, total = 0, 0, 0
            for x1, x2, y in loader:
                x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
                with torch.set_grad_enabled(train):
                    logits = model(x1, x2)
                    loss = criterion(logits, y)
                    if train:
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * y.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
            return correct / total

        val_report = {}
        for epoch in tqdm(range(epochs), desc="Training GRU..."):
            tr_acc = run_epoch(tr_loader, True)
            val_acc = run_epoch(val_loader, False)
            val_report[epoch] = [tr_acc, val_acc]
            print(f"Epoch {epoch+1}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
        
        save_json(val_rep_path, val_report)
        torch.save(model, trained_model_path)
        logging.info("Exited Method train_gru")

    def train(self):
        logging.info("Entered Method train")
        
        if self.config.own_emb_model:
            emb_q1 = np.load(self.config.q1_emb_own)
            emb_q2 = np.load(self.config.q2_emb_own)
            trained_model_path = self.config.trained_model_with_own_emb
            val_rep_path = Path(self.config.own_val_report)
        else:
            emb_q1 = np.load(self.config.q1_emb_bert)
            emb_q2 = np.load(self.config.q2_emb_bert)
            trained_model_path = self.config.trained_model_with_bert_emb
            val_rep_path = Path(self.config.bert_val_report)
            
        y = pd.read_csv(self.config.y_path).values.flatten()

        self.train_gru(emb_q1, emb_q2, y, trained_model_path, val_rep_path)
        logging.info("Exited Method train")