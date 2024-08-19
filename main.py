import os
from text_recognizer import TextRecognizer

recognizer = TextRecognizer()

try:
    recognizer.process()
except FileNotFoundError:
    os.rmdir('train_data')
    recognizer.process()
