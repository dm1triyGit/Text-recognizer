import numpy as nmp
import os
import csv

from numpy.ma.core import argmax

from network import NeuralNetwork
from config.network_config import *
from config.recognizer_config import *


class TextRecognizer:
    def __init__(self):
        self.__network = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    def process(self):
        if os.path.exists(TRAIN_DATA_DIRECTORY):
            input_hidden_weights = []
            hidden_output_weights = []
            with open(TRAIN_DATA_WIH_FILENAME, "r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    input_hidden_weights.append(nmp.asarray(row, dtype=nmp.float32))

            with open(TRAIN_DATA_WHO_FILENAME, "r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    hidden_output_weights.append(nmp.asarray(row, dtype=nmp.float32))

            self.__network.seed_weights(nmp.asarray(input_hidden_weights), nmp.asarray(hidden_output_weights))

        else:
            input_hidden_weights, hidden_output_weights = self.__train_network()

            os.mkdir(TRAIN_DATA_DIRECTORY)
            self.__write_trained_data(input_hidden_weights, hidden_output_weights)

        self.__test_network()

    def recognize_image(self, inputs_array):
        img_data = 255.0 - inputs_array
        img_data = (img_data / 255.0 * 0.99) + 0.01
        return argmax(self.__network.query(img_data))

    def retrain_network(self, inputs_array, correct_answer):
        img_data = 255.0 - inputs_array
        inputs = (img_data / 255.0 * 0.99) + 0.01
        targets = nmp.zeros(OUTPUT_NODES) + 0.01
        targets[correct_answer] = 0.99
        for e in range(EPOCHS * INPUT_NODES):
            self.__network.train(inputs, targets)

        input_hidden_weights, hidden_output_weights = self.__network.get_train_data()
        self.__write_trained_data(input_hidden_weights, hidden_output_weights)

    def __write_trained_data(self, input_hidden_weights, hidden_output_weights):
        with open(TRAIN_DATA_WIH_FILENAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(input_hidden_weights)

        with open(TRAIN_DATA_WHO_FILENAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(hidden_output_weights)

    def __train_network(self):
        with open(TRAIN_DATA_FILENAME, 'r') as training_data_file:
            training_data_list = training_data_file.readlines()

        for e in range(EPOCHS):
            for record in training_data_list:
                all_val = record.split(',')
                inputs = ((nmp.asarray(all_val[1:], dtype=nmp.float32)) / 255.0 * 0.99) + 0.01
                targets = nmp.zeros(OUTPUT_NODES) + 0.01
                targets[int(all_val[0])] = 0.99

                self.__network.train(inputs, targets)

        return self.__network.get_train_data()

    def __test_network(self):
        test_data_file = open(TEST_DATA_FILENAME, 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        scorecards = []
        for record in test_data_list:
            all_val = record.split(',')
            correct_label = int(all_val[0])

            inputs = ((nmp.asarray(all_val[1:], dtype=nmp.float32)) / 255.0 * 0.99) + 0.01
            outputs = self.__network.query(inputs)
            index_max_output = nmp.argmax(outputs)

            if index_max_output == correct_label:
                scorecards.append(1)
            else:
                scorecards.append(0)

        scorecards_array = nmp.asarray(scorecards)
        network_rate = (scorecards_array.sum() / scorecards_array.size) * 100
        print('Эффективность сети -', network_rate, '%')
