import nltk
import random
from nltk.corpus import words as nltk_words
import logging
import random

def download_nltk_words():
    try:
        # Check if the "words" dataset is already downloaded
        nltk.data.find('corpora/words')
        logger = logging.getLogger()
    except LookupError:
        # If not found, download it
        nltk.download("words")

def generate_run_name():
    random.seed()
    download_nltk_words()
    logger = logging.getLogger()
    english_words = [word for word in nltk_words.words() if len(word) == 4]  # Filter for 4-letter words
    random_words = random.sample(english_words, 2)
    run_name = " ".join(random_words)
    run_name = run_name.title()  # Capitalize the first letter of each word
    logger.info('Run name is: ' + run_name)
    return run_name