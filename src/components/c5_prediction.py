from src.logger import logging
from src.entities.config_entity import PredictionConfig
import pickle
import re
import contractions
import tqdm
from tqdm.auto import tqdm
import yaml
from symspellpy import SymSpell
import pkg_resources
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Prediction:
    def __init__(self, config: PredictionConfig):
        logging.info("Prediction class Initialization started")
        self.config = config
        logging.info("Prediction class Initialization completed")
    
    def preprocess(self, text):
        logging.info("Entered method preprocess")
        text = text.lower()                                        #lower case
        text = re.sub(r'<.*?>',' ',text)               #remove html tags
        text = re.sub(r'http\S+|www\.\S+',' ',text)    #remove urls
        
        # expand contractions
        for contraction, expanded in tqdm(contractions.contractions_dict.items(), desc="Expanding Contractions..."):
            text = re.sub(rf'\b{re.escape(contraction.lower())}\b', expanded.lower(), text)
        
        # replace chatwords with full words
        chatword_path = self.config.chatwords_file
        with open(chatword_path, "r", encoding="utf-8") as f:
            chat_words = yaml.safe_load(f)
        for chat_word, full_word in tqdm(chat_words.items(), desc="Expanding Chatwords..."):
            text = re.sub(rf'\b{re.escape(chat_word.lower())}\b', full_word, text)
        
        # replace symbols
        sym_replace = {
            '%': ' percent ',
            '$': ' dollar ',
            '₹': ' rupee ',
            '₹': ' rupee ',
            '€': ' euro ',
            '@': ' at '
        }
        for sym, replacement in tqdm(sym_replace.items(), desc="Replacing Symbols..."):
            text = re.sub(rf'\b{re.escape(sym)}\b', replacement, text)
        
        # remove stopwords
        #stop_words = stopwords.words('english')
        #for stop_word in tqdm(stop_words, desc='Removing Stopwords...'):
        #    text = text.replace(rf'\b{re.escape(stop_word)}\b', ' ', regex=True)
        
        # remove emojis
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        text = re.sub(emoji_pattern, ' ', text)
        
        # remove punctuations
        text = re.sub(r'''[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]''',' ',text)
        
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # correct spellings
        if not isinstance(text, str) or not text.strip():
            text = ''
        else:
            sym_spell = SymSpell(max_dictionary_edit_distance=2)
            dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
            sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
            if suggestions:
                text = suggestions[0].term
    
        logging.info("Exited method preprocess")
        return text
    
    def processing_with_self_emb_model(self, text):
        logging.info("Entered method processing_with_self_emb_model")
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokenized_text = text.split()
        with open(self.config.word2idx_path, 'rb') as f:
            word2idx = pickle.load(f)
        encoded_text = [word2idx.get(token, 1) for token in tokenized_text]
        embedding_matrix = torch.load(self.config.emb_matrix, weights_only=False)
        if not encoded_text:
            text_emb = torch.zeros(embedding_matrix.shape[1])
        else:
            text_emb = torch.mean(embedding_matrix[encoded_text], dim=0)
        # reshape to (1, seq_len, input_size) to match GRU training input
        seq_len = self.config.seq_len
        text_emb = text_emb.unsqueeze(0).reshape(1, seq_len, -1)
        logging.info("Exited method processing_with_self_emb_model")
        return text_emb
    
    def processing_with_bert_emb_model(self, text):
        logging.info("Entered method processing_with_bert_emb_model")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        text_emb = torch.tensor(model.encode(text))
        # reshape to (1, seq_len, input_size) to match GRU training input
        seq_len = self.config.seq_len
        text_emb = text_emb.unsqueeze(0).reshape(1, seq_len, -1)
        logging.info("Exited method processing_with_bert_emb_model")
        return text_emb
    
    def process(self, text):
        logging.info("Entered method process")
        if self.config.own_emb_model:
            text_emb = self.processing_with_self_emb_model(text)
        else:
            text_emb = self.processing_with_bert_emb_model(text)
        logging.info("Exited method process")
        return text_emb
        
    
    def predict(self, q1, q2):
        logging.info("Entered method predict")
        q1_preprocessed, q2_preprocessed = self.preprocess(q1), self.preprocess(q2)
        q1_emb, q2_emb = self.process(q1_preprocessed), self.process(q2_preprocessed)
        # Load the trained model
        if self.config.own_emb_model:
            model = torch.load(self.config.trained_model_with_own_emb, weights_only=False)
        else:
            model = torch.load(self.config.trained_model_with_bert_emb, weights_only=False)
        
        model.eval()
        with torch.no_grad():
            q1_emb, q2_emb = q1_emb.to(DEVICE), q2_emb.to(DEVICE)
            prediction = model(q1_emb, q2_emb)
            is_duplicate = prediction > 0.4
        logging.info("Exited method predict")
        return is_duplicate.to("cpu")