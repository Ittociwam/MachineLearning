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
    def __init__(self, learning_rate, data, targets, inputs, num_layers=2):
        self.learning_rate = learning_rate
        self.data = data
        self.bias = -1
        self.targets = targets
        self.labels = np.unique(targets)
        self.inputs = inputs
        self.neurons = []
        self.num_inputs = len(self.data.columns)
        self.input_list = []
        self.layers = []
        #self.label_dict = defaultdict(list)
        print("data_inputs: \n", self.data)
        self.build_network(num_layers)

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
                num_neurons = 2
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
            outputs = []
            iteration_inputs = data_inputs
            for layer in (self.layers):
                outputs = []
                for neuron in layer:
                    counter += 1
                    outputs.append(neuron.compute_activation(iteration_inputs))
                iteration_inputs = (1, outputs)
            #error = sum_of_squares_error(outputs) #will compare with targets
            #go_back_and_change_weights(error)
            network_output.append(outputs)
        print(network_output, "COUNTER: ", counter)


    class Neuron:
        def __init__(self, num_inputs):
            self.rangemax = 1.0
            self.rangemin = -1.0
            self.inputs = []
            self.threshold = 0
            for i in range(num_inputs):
                print("appending input number: ", i)
                self.inputs.append(self.Input(random.uniform(self.rangemin, self.rangemax)))
            #make bias input the last input
            self.inputs.append(self.Input(random.uniform(self.rangemin, self.rangemax)))
            print("making a neuron with ", num_inputs, " inputs\n")

        def compute_activation(self, input):
            #assert input is len(self.inputs)
            _sum = 0
            #loop through all inputs
            #print("for compute we are looping through range: ", range(len(self.inputs) - 1))
            for i in range(len(self.inputs) -1):
                _sum += self.inputs[i].weight * input[1][i]
            #calculate for bias input -1 gets last element in an array
            _sum += self.inputs[-1].weight * 1
            self.sigmoid(_sum)
            return _sum #> self.threshold:
               # return 1
           # else:
              #  return 0

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-1))

        def dsigmoid(self, y):
            return y ( (1.0 - y))

        class Input:
            def __init__(self, weight):
                self.weight = weight




print("**********************CSV**************************")

my_dataset = Dataset('indian', .3, 3333, False)

nn = NeuralNetwork(.3, my_dataset.data_train, my_dataset.target_train, my_dataset.data_test)

nn.train_network()


#closest = nearestneighbor.knn()

#closest = closest.astype(int)

#print("closest: ", closest)

#acc_score = accuracy_score(my_dataset.target_test, closest)

#print("Accuracy of .knn() is: ", acc_score)

