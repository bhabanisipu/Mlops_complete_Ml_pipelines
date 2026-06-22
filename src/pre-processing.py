import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transforms the input text by performing tokenization, stop word removal, and stemming."""
    ps = PorterStemmer()

    text = text.lower()

    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = [ps.stem(word) for word in text]

    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """preprocesses the input DataFrame by transforming the text column and encoding the target column."""
    try:
        logger.debug("starting the processing of Dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("target column encoded")

        df = df.drop_duplicates(keep='first')
        logger.debug("duplicates are removed")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("text column transformed")
        return df
    except KeyError as e:
        logger.error('column not found: %s',e)
        raise
    except Exception as e:
        logger.error('error during text normalization: %s',e)
        raise
def main(text_column='text',target_column='target'):
    """main function is to load raw data ,preprocess it, and save the process data."""
    try:
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug('data loaded successfully')

        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        data_path = os.path.join('./data',"interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index = False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index = False)

        logger.debug('processed data saved to: %s',data_path)
    except FileNotFoundError as e:
        logger.error("file not found: %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("no data: %s",e)
        raise
    except Exception as e:
        logger.error("failed to complete the data transformation process: %s",e)
        raise

if __name__ == '__main__':
    main()
