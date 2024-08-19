import os
from text_recognizer import TextRecognizer
from config.recognizer_config import *

recognizer = TextRecognizer()

try:
    recognizer.process()
except FileNotFoundError:
    os.rmdir(TRAIN_DATA_DIRECTORY)
    recognizer.process()
