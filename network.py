import numpy as nmp
import scipy.special as spc


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.__inodes = inputnodes
        self.__hnodes = hiddennodes
        self.__onodes = outputnodes
        self.__lr = learningrate

        self.__wih = (nmp.random.normal(0.0, pow(self.__hnodes, -0.5), (self.__hnodes, self.__inodes)))
        self.__who = (nmp.random.normal(0.0, pow(self.__onodes, -0.5), (self.__onodes, self.__hnodes)))

        self.__activation_func = lambda x: spc.expit(x)

    def train(self, inputs_list, target_list):
        inputs = nmp.array(inputs_list, ndmin=2).T
        targets = nmp.array(target_list, ndmin=2).T
        hidden_inputs = nmp.dot(self.__wih, inputs)
        hidden_outputs = self.__activation_func(hidden_inputs)

        final_inputs = nmp.dot(self.__who, hidden_outputs)
        final_outputs = self.__activation_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = nmp.dot(self.__who.T, output_errors)

        self.__who += self.__lr * nmp.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), nmp.transpose(hidden_outputs))
        self.__wih += self.__lr * nmp.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                          nmp.transpose(inputs))

    def query(self, inputs_list):
        inputs = nmp.array(inputs_list, ndmin=2).T

        hidden_inputs = nmp.dot(self.__wih, inputs)
        hidden_outputs = self.__activation_func(hidden_inputs)

        final_inputs = nmp.dot(self.__who, hidden_outputs)
        final_outputs = self.__activation_func(final_inputs)

        return final_outputs

    def seed_weights(self, weight_input_hidden, weight_hidden_output):
        self.__wih = weight_input_hidden
        self.__who = weight_hidden_output

    def get_train_data(self):
        return self.__wih, self.__who
