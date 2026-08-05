import cudf.pandas
cudf.pandas.install()
import pandas as pd
import yaml
from src.entities.config_entity import DataPreProcessingConfig
from src.logger import logging
import re
import contractions
import tqdm
from tqdm.auto import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from symspellpy import SymSpell
import pkg_resources

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        logging.info("DataPreProcessing class Initialization started")
        self.config = config
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
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
        df['question1'] = df['question1'].fillna('')
        df['question2'] = df['question2'].fillna('')

        def correct_text(text):
            if not isinstance(text, str) or not text.strip():
                return ''
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestions[0].term if suggestions else text

        tqdm.pandas(desc="Correcting Spelling q1...")
        df['question1'] = df['question1'].progress_apply(correct_text)
        tqdm.pandas(desc="Correcting Spelling q2...")
        df['question2'] = df['question2'].progress_apply(correct_text)
        logging.info("Exited Method correct_spelling")
        return df

    def save_X_y(self, df):
        logging.info("Entered Method save_y")
        df[['question1','question2']].to_csv(self.config.X_path, index=False)
        df['is_duplicate'].to_csv(self.config.y_path, index=False)
        df.to_csv(self.config.preprocessed_data, index=False)
        logging.info("Exited Method save_y")
    
    def preprocess(self):
        logging.info("Entered Method preprocess")
        df = pd.read_csv(self.config.raw_data_file)
        
        # preprocessing
        df = self.remove_unnecessary_cols(df)
        df = self.lower_case(df)
        df = self.remove_html_tags(df)
        df = self.remove_web_links(df)
        df = self.expand_contractions(df)
        df = self.expand_chatwords(df)
        df = self.replace_symbols(df)
        ####df = self.remove_stopwords(df)
        df = self.remove_emojis(df)
        df = self.remove_punctuations(df)
        df = self.remove_extra_whitespace(df)
        df = self.correct_spelling(df)
        
        # save preprocessed csv
        self.save_X_y(df)
        logging.info("Exited Method preprocess")