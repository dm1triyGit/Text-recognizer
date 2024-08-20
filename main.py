import os
from text_recognizer import TextRecognizer
from config.recognizer_config import *
from my_canvas import MyCanvas

recognizer = TextRecognizer()
canvas = MyCanvas(recognizer.recognize_image, recognizer.retrain_network)

try:
    recognizer.process()
    canvas.show()
except FileNotFoundError:
    os.rmdir(TRAIN_DATA_DIRECTORY)
    recognizer.process()
