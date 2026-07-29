#import cudf.pandas
#cudf.pandas.install()
import pandas as pd

import numba
import yaml
from src.entities.config_entity import DataPreProcessingConfig
from src.logger import logging
import re
import contractions
import tqdm
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import pkg_resources

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        logging.info("DataPreProcessing class Initialization started")
        self.config = config
        logging.info("DataPreProcessing class Initialization completed")
    
    def remove_unnecessary_cols(self, df):
        logging.info("Entered Method remove_unnecessary_cols")
        df.drop(columns=['id','qid1','qid2'], axis=1, inplace=True)
        logging.info("Exited Method remove_unnecessary_cols")
        return df
    
    def lower_case(self, df):
        logging.info("Entered Method lower_case")
        df['question1'] = df['question1'].str.lower()
        df['question2'] = df['question2'].str.lower()
        logging.info("Exited Method lower_case")
        return df
    
    def remove_html_tags(self, df):
        logging.info("Entered Method remove_html_tags")
        df['question1'] = df['question1'].str.replace(r'<.*?>',' ',regex=True)
        df['question2'] = df['question2'].str.replace(r'<.*?>',' ',regex=True)
        logging.info("Exited Method remove_html_tags")
        return df
    
    def remove_web_links(self, df):
        logging.info("Entered Method remove_web_links")
        df['question1'] = df['question1'].str.replace(r'http\S+|www\.\S+',' ',regex=True)
        df['question2'] = df['question2'].str.replace(r'http\S+|www\.\S+',' ',regex=True)
        logging.info("Exited Method remove_web_links")
        return df
    
    def expand_contractions(self, df):
        logging.info("Entered Method expand_contractions")
        question1 = df['question1']
        question2 = df['question2']
        for contraction, expanded in tqdm(contractions.contractions_dict.items(), desc="Expanding Contractions..."):
            question1 = question1.str.replace(rf'\b{re.escape(contraction.lower())}\b', expanded.lower(), regex=True)
            question2 = question2.str.replace(rf'\b{re.escape(contraction.lower())}\b', expanded.lower(), regex=True)
        df['question1'] = question1
        df['question2'] = question2
        logging.info("Exited Method expand_contractions")
        return df
    
    def expand_chatwords(self, df):
        logging.info("Entered Method expand_chatwords")
        chatword_path = self.config.chatwords_file
        with open(chatword_path, "r", encoding="utf-8") as f:
            chat_words = yaml.safe_load(f)

        question1 = df['question1']
        question2 = df['question2']
        for chat_word, full_word in tqdm(chat_words.items(), desc="Expanding Chatwords..."):
            question1 = question1.str.replace(rf'\b{re.escape(chat_word.lower())}\b', full_word, regex=True)
            question2 = question2.str.replace(rf'\b{re.escape(chat_word.lower())}\b', full_word, regex=True)
        df['question1'] = question1
        df['question2'] = question2
        logging.info("Exited Method expand_chatwords")
        return df
    
    def replace_symbols(self, df):
        logging.info("Entered Method replace_symbols")
        sym_replace = {
            '%': ' percent ',
            '$': ' dollar ',
            '₹': ' rupee ',
            '₹': ' rupee ',
            '€': ' euro ',
            '@': ' at '
        }
        q1 = df['question1']
        q2 = df['question2']
        for sym, replacement in tqdm(sym_replace.items(), desc="Replacing Symbols..."):
            q1 = q1.str.replace(rf'\b{re.escape(sym)}\b', replacement, regex=True)
            q2 = q2.str.replace(rf'\b{re.escape(sym)}\b', replacement, regex=True)
        df['question1'] = q1
        df['question2'] = q2
        logging.info("Exited Method replace_symbols")
        return df
    
    def remove_stopwords(self, df):
        logging.info("Entered Method remove_stopwords")
        stop_words = stopwords.words('english')
        stopword_removed_q1 = df['question1']
        stopword_removed_q2 = df['question2']
        for stop_word in tqdm(stop_words, desc='Removing Stopwords...'):
            stopword_removed_q1 = stopword_removed_q1.str.replace(rf'\b{re.escape(stop_word)}\b', ' ', regex=True)
            stopword_removed_q2 = stopword_removed_q2.str.replace(rf'\b{re.escape(stop_word)}\b', ' ', regex=True)
        df['question1'] = stopword_removed_q1
        df['question2'] = stopword_removed_q2
        logging.info("Exited Method remove_stopwords")
        return df
    
    def remove_emojis(self, df):
        logging.info("Entered Method remove_emojis")
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        df['question1'] = df['question1'].str.replace(emoji_pattern, ' ', regex=True)
        df['question2'] = df['question2'].str.replace(emoji_pattern, ' ', regex=True)
        logging.info("Exited Method remove_emojis")
        return df
    
    def remove_punctuations(self, df):
        logging.info("Entered Method remove_punctuations")
        df['question1'] = df['question1'].str.replace(r'''[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]''',' ',regex=True)
        df['question2'] = df['question2'].str.replace(r'''[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]''',' ',regex=True)
        logging.info("Exited Method remove_punctuations")
        return df
    
    def remove_extra_whitespace(self, df):
        logging.info("Entered Method remove_extra_whitespace")
        df['question1'] = df['question1'].str.strip().str.replace(r'\s+',' ',regex=True)
        df['question2'] = df['question2'].str.strip().str.replace(r'\s+',' ',regex=True)
        logging.info("Exited Method remove_extra_whitespace")
        return df
    
    def correct_spelling(self, df):
        logging.info("Entered Method correct_spelling")
        # correct spellings using symspellpy
        tokens_series1 = df['question1'].str.findall(r'\b[a-z]+\b')
        tokens_series2 = df['question2'].str.findall(r'\b[a-z]+\b')
        tokens_series = pd.concat([tokens_series1,tokens_series2], axis=0, ignore_index=True) #tokenization per row

        all_words = tokens_series.explode().dropna()   # all unique words
        unique_words = all_words.unique()#.to_pandas()  # CPU transfer

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        sym.load_dictionary(dict_path, term_index=0, count_index=1)

        # creating spell correction map (dictionary) for current data
        correction_map = {}
        for word in unique_words:
            if word not in sym.words:  # word not in dictionary → likely misspelled
                suggestions = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    correction_map[word] = suggestions[0].term

        # correcting the spellings
        corrected_q1 = df['question1']
        corrected_q2 = df['question2']
        for wrong, right in tqdm(correction_map.items(), desc="Correcting Questions"):
            corrected_q1 = corrected_q1.str.replace(rf'\b{re.escape(wrong)}\b', right, regex=True)
            corrected_q2 = corrected_q2.str.replace(rf'\b{re.escape(wrong)}\b', right, regex=True)
        df['question1'] = corrected_q1
        df['question2'] = corrected_q2
        logging.info("Exited Method correct_spelling")
        return df
    
    def preprocess(self):
        logging.info("Entered Method preprocess")
        raw_df_path = self.config.raw_data_file
        preprocessed_df_path = self.config.preprocessed_data
        # load raw csv
        df = pd.read_csv(raw_df_path)
        
        # preprocessing
        df = self.remove_unnecessary_cols(df)
        df = self.lower_case(df)
        df = self.remove_html_tags(df)
        df = self.remove_web_links(df)
        df = self.expand_contractions(df)
        df = self.expand_chatwords(df)
        df = self.replace_symbols(df)
        #df = self.remove_stopwords(df)
        df = self.remove_emojis(df)
        df = self.remove_punctuations(df)
        df = self.remove_extra_whitespace(df)
        df = self.correct_spelling(df)
        
        # save preprocessed csv
        df.to_csv(preprocessed_df_path, index=False)
        logging.info("Exited Method preprocess")