import nltk
import random
from nltk.corpus import words as nltk_words
import logging

nltk.download("words")

def generate_run_name():
    logger = logging.getLogger()
    english_words = nltk_words.words()
    random_words = random.sample(english_words, 2)
    run_name = " ".join(random_words)
    run_name = run_name.title()  # Capitalize the first letter of each word
    logger.info(run_name)
    return run_name