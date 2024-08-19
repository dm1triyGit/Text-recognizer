import numpy as nmp
import os
import csv

from network import NeuralNetwork
from config.network_config import *


class TextRecognizer:
    def __init__(self):
        self.__network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        pass

    def process(self):
        if os.path.exists('train_data'):
            input_hidden_weights = []
            hidden_output_weights = []
            with open('train_data/wih.csv', "r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    input_hidden_weights.append(nmp.asarray(row, dtype=nmp.float32))

            with open('train_data/who.csv', "r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    hidden_output_weights.append(nmp.asarray(row, dtype=nmp.float32))

            self.__network.seed_weights(nmp.asarray(input_hidden_weights), nmp.asarray(hidden_output_weights))
            pass
        else:
            input_hidden_weights, hidden_output_weights = self.__train_network()

            os.mkdir('train_data')
            with open('train_data/wih.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(input_hidden_weights)

            with open('train_data/who.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(hidden_output_weights)

        self.__test_network()
        pass

    def __train_network(self):
        with open('assets/mnist_train.csv', 'r') as training_data_file:
            training_data_list = training_data_file.readlines()

        for e in range(epochs):
            for record in training_data_list:
                all_val = record.split(',')
                inputs = ((nmp.asarray(all_val[1:], dtype=nmp.float32)) / 255.0 * 0.99) + 0.01
                targets = nmp.zeros(output_nodes) + 0.01
                targets[int(all_val[0])] = 0.99

                self.__network.train(inputs, targets)
                pass

            pass

        return self.__network.get_train_data()

    def __test_network(self):
        test_data_file = open('assets/mnist_test.csv', 'r')
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
            pass

        scorecards_array = nmp.asarray(scorecards)
        network_rate = (scorecards_array.sum() / scorecards_array.size) * 100
        print('Эффективность сети -', network_rate, '%')

        pass


pass
