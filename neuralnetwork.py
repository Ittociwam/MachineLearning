import random

import numpy as np
from scipy.stats import mode
from sklearn.cross_validation import train_test_split
from scipy.stats import entropy as entropy
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict
import sys


class Dataset:
    def __init__(self, csv, test_size, random_state, convert_nominal=False):

        #Read in the csv
        self.ds = pd.read_csv(csv, header=None)
        #get the  column length
        ds_num_col = len(self.ds.columns)

        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.std_train = []
        self.std_test = []

        #print("The dataframe before: \n", self.ds)
        #if the user wants nominal data, divide any columns without strings to 4 buckets and assign them nominal values (0, 1, 2, 3)
        if convert_nominal is True:
            self.convert_numerical_data(self.ds)
        else: #otherwise convert any string data to numbers
            self.convert_nominal_data(self.ds)
       # print("The dataframe after: \n", self.ds)

        self.data = self.ds.loc[:, : ds_num_col - 2]
        self.targets = self.ds[ds_num_col - 1]

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
        # standardize the data if we are using numerical data
        if not convert_nominal:
            self.standardize_data()
        self.std_train = [0] * int(len(self.data_train))
        self.std_test = [0] * int(len(self.data_test))

       #print("train data and targets", self.data_train, self.target_train)

    # This function turns nominal data into number values that are not scaled
    def convert_nominal_data(self, dataset):
        le = preprocessing.LabelEncoder()
        num_col = len(dataset.columns)

        for i in range(0, num_col):
            has_string = False
            le.fit(dataset[i])
            list_of_classes = list(le.classes_)
            for a_class in list_of_classes:
                if isinstance(a_class, str):
                    has_string = True
            if has_string:
                dataset[i] = le.transform(dataset[i])

    # this function turns numeric data into generalized data (ex. 1-2 = 0, 3-4 = 1, 5-6 = 2)
    def convert_numerical_data(self, dataset):
        num_col = len(dataset.columns)
        le = preprocessing.LabelEncoder()

        for i in range(0, num_col):
            nominal_col = False
            le.fit(dataset[i])
            list_of_classes = list(le.classes_)
            for a_class in list_of_classes:
               if isinstance(a_class, str):
                  nominal_col = True
                  break
            if nominal_col:
               continue
            else:
               dataset[i] = pd.qcut(dataset[i], 4, labels=False)



    # This function uses a zscore to standardize the data
    def standardize_data(self):
        # fill std_train with standardized values
        self.std_train = (self.data_train - self.data_train.mean()) / self.data_train.std()
        # fill std_test with standardized values
        self.std_test = (self.data_test - self.data_train.mean()) / self.data_train.std()



class NeuralNetwork:
    def __init__(self, learning_rate, data, targets, num_layers=1):
        self.learning_rate = learning_rate
        self.data = data
        self.bias = -1
        self.targets = targets
        self.labels = np.unique(targets)
        self.neurons = []
        self.num_inputs = len(self.data.columns)
        self.input_list = []
        self.layers = []
        #self.label_dict = defaultdict(list)
        print("data_inputs: \n", self.data)
        print("targets: \n", targets)
        self.build_network(num_layers)
        self.TARGET_CODES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #                      a       b
        #self.TARGET_CODES = [[1, 0], [0, 1]]

    # build_network will build the network, the network will be stored in a multidimentional array
    # A network with 3 input neurons, 1 layer of 2 hidden neurons and 4 output neurons would be stored like this:
    # [[Neuron, Neuron, Neuron], [Neuron, Neuron], [Neuron, Neuron, Neuron, Neuron]]
    # The initial layer will be built with the number of inputs of the dataset and the rest will be built with the
    # number of neurons from the previous layer
    def build_network(self, num_layers):
        #The initial inputs for layer 1 will be the number of inputs of the data
        layer_inputs = self.num_inputs
        # if the user has specified more than 1 layer
        if num_layers > 1:
            #loop through the layers except the last layer which we will handle separately
            for i in range(num_layers - 1):
                this_layer = []
                #prompt user for num neurons in this layer
                prompt = "How many neurons in layer " + str(i) + " ?  > "
                num_neurons = 2 #input(prompt)
                num_neurons = int(num_neurons)
                for j in range(num_neurons):
                    #make that many neurons in the layer
                    new_neuron = self.Neuron(layer_inputs)
                    this_layer.append(new_neuron)
                #grab the number of neurons to use as layer inputs for our next layer
                layer_inputs = num_neurons
                #append that layer to the NN layers array
                self.layers.append(this_layer)
        # this will either be the last layer of our MLP or a single layer
        last_layer = []
        for label in self.labels:
            new_neuron = self.Neuron(layer_inputs)
            last_layer.append(new_neuron)
        self.layers.append(last_layer)

    def train_network(self):
        network_output = []
        counter = 0
        #loop through each row
        for data_inputs in self.data.iterrows():
            layer_outputs = []
            iteration_inputs = data_inputs
            all_outputs = []
            for layer in (self.layers):
                layer_outputs = []
                for neuron in layer:
                    counter += 1
                    layer_outputs.append(neuron.compute_activation(iteration_inputs))
                iteration_inputs = (1, layer_outputs)
                all_outputs.append(layer_outputs)
            is_output = True
            #iterate backwards thru the layers starting with output layer
            prev_error = None
            for i in range(len(self.layers) - 1, -1, -1):
                if is_output:
                    error = self.error_output(self.layers[i], all_outputs[i], self.targets[data_inputs[0]])
                    self.add_new_weights(self.layers[i], all_outputs[i], error)
                    prev_error = error
                    is_output = False
                    #calc weight updates?
                else:
                    error = self.error_hidden(self.layers[i], all_outputs[i], self.layers[i + 1], prev_error)
                    prev_error = error
                    self.add_new_weights(self.layers[i], all_outputs[i], error)
            self.update_weights()
            network_output.append(layer_outputs)


    def add_new_weights(self, layer, outputs, errors):
        for i, neuron in enumerate(layer):
            for input in neuron.neuron_inputs:
                input.new_weight = input.weight - (self.learning_rate*outputs[i]*errors[i])

    def error_output(self, layer, outputs, target):
        target_code = self.TARGET_CODES[target]
        error_list = []
        #output layer outputs are in pandas format
        for i in range(0, len(layer)):
            error = (outputs[i]) * (1 - outputs[i]) * (outputs[i] - target_code[i])
            error_list.append(error)
        return error_list



    def error_hidden(self, layer, outputs, prev_layer, error_list):
        _sum_weights_errors = 0
        for i in range(0, len(layer)):
            for j, neuron in enumerate(prev_layer):
                #WHAT INPUTS ARE GOING IN HERE?
                _sum_weights_errors += neuron.neuron_inputs[i].weight * error_list[j]
            error = (outputs[i]) * (1-outputs[i]) * _sum_weights_errors
            error_list.append(error)
        return error_list


    def update_weights(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.update_weights()

    def feed_forward(self, inputs):
        prediction_outputs = []
        #run the test data through the network and send back an array of predicted targets
        for test_input in inputs.iterrows():
            outputs = []
            iteration_inputs = test_input
            for layer in self.layers:
                outputs = []
                for neuron in layer:
                    outputs.append(neuron.compute_activation(iteration_inputs))
                iteration_inputs = (1, outputs)
            prediction_outputs.append(outputs)
        return prediction_outputs

    def predict(self, inputs):
        prediction_outputs = self.feed_forward(inputs)
        # prediction_outputs = []
        # #run the test data through the network and send back an array of predicted targets
        # for test_input in inputs.iterrows():
        #     outputs = []
        #     iteration_inputs = test_input
        #     for layer in self.layers:
        #         outputs = []
        #         for neuron in layer:
        #             outputs.append(neuron.compute_activation(iteration_inputs))
        #         iteration_inputs = (1, outputs)
        #     prediction_outputs.append(outputs)
        prediction = self.get_prediction(prediction_outputs)
        return prediction

    def get_prediction(self, prediction_outputs):
        prediction = []
        for prediction_output in prediction_outputs:
            prediction.append(np.argmax(prediction_output))
        return prediction





    class Neuron:
        def __init__(self, num_inputs):
            self.rangemax = 1.0
            self.rangemin = -1.0
            self.neuron_inputs = []
            self.threshold = 0
            #self.testweight = 0.1
            for i in range(num_inputs):
                #self.testweight += 0.1

                weight = random.uniform(self.rangemin, self.rangemax)
                print("appending input number: ", i, "with weight: ", weight)
                self.neuron_inputs.append(self.Input(weight))
                #self.neuron_inputs.append(self.Input(self.testweight))
            #make bias input the last input
            print("bias: ")
            bias_weight = random.uniform(self.rangemin, self.rangemax)
            self.neuron_inputs.append(self.Input(bias_weight))
            #self.neuron_inputs.append(self.Input(self.testweight + .1))
            print("making a neuron with ", num_inputs, " inputs\n")

        def update_weights(self):
            for input in self.neuron_inputs:
                input.update_weight()

        def compute_activation(self, input):
            #assert input is len(self.neuron_inputs)
            _sum = 0
            #loop through all inputs
            for i in range(len(self.neuron_inputs) -1):
                _sum += self.neuron_inputs[i].weight * input[1][i]
            #calculate for bias input -1 gets last element in an array
            _sum += self.neuron_inputs[-1].weight * -1
            return self.sigmoid(_sum) #> self.threshold:
               # return 1
           # else:
              #  return 0

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(self, y):
            return y ( (1.0 - y))

        class Input:
            def __init__(self, weight):
                print("weight: ", weight)
                self.weight = weight
                self.new_weight = None

            def update_weight(self):
                self.weight = self.new_weight




print("**********************CSV**************************")

my_dataset = Dataset('indian', .3, 3333, False)

nn = NeuralNetwork(.3, my_dataset.data_train, my_dataset.target_train, 2)

for i in range(15):
    nn.train_network()

prediction = nn.predict(my_dataset.data_test)


#closest = nearestneighbor.knn()

#closest = closest.astype(int)

#print("closest: ", closest)

acc_score = accuracy_score(my_dataset.target_test, prediction)

print("Accuracy of Neural Network is: ", acc_score)

