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
    def __init__(self, learning_rate, data, targets, inputs, bias=-1):
        self.learning_rate = learning_rate
        self.data = data
        self.bias = bias
        self.targets = targets
        self.labels = np.unique(targets)
        self.inputs = inputs
        self.neurons = []
        num_inputs = len(self.inputs.columns)
        self.input_list = []
        self.label_dict = defaultdict(list)
        print("inputs: \n", self.inputs)
        for label in self.labels:
            new_neuron = self.Neuron(num_inputs, label)
            self.neurons.append(new_neuron)


    def train_network(self):
        outputs = []
        #loop through each row
        for row in self.data.iterrows():
            print("row calculating: ", row)
            output = []
            for neuron in self.neurons:
                output.append(neuron.compute_activation(row))
            outputs.append(output)
        print(outputs)



    class Neuron:
        def __init__(self, num_inputs, label):
            self.rangemax = 1.0
            self.rangemin = -1.0
            self.inputs = []
            self.label = label
            self.threshold = 0
            for i in range(num_inputs):
                self.inputs.append(self.Input(i,random.uniform(self.rangemin, self.rangemax) ))
            #make bias input the last input
            self.inputs.append(self.Input(num_inputs + 1, random.uniform(self.rangemin, self.rangemax)))
            print("making a neuron")

        def compute_activation(self, input):
            _sum = 0
            #loop through all inputs
            for i in range(len(self.inputs) - 1):
                _sum += self.inputs[i].weight * input[1][i]
            #calculate for bias input -1 gets last element in an array
            _sum += self.inputs[-1].weight * 1
            if _sum > self.threshold:
                return 1
            else:
                return 0

        class Input:
            def __init__(self, attribute, weight):
                self.attribute = attribute
                self.weight = weight




print("**********************CSV**************************")

my_dataset = Dataset('iris.csv', .3, 3333, False)

nn = NeuralNetwork(.3, my_dataset.data_train, my_dataset.target_train, my_dataset.data_test)

nn.train_network()


#closest = nearestneighbor.knn()

#closest = closest.astype(int)

#print("closest: ", closest)

#acc_score = accuracy_score(my_dataset.target_test, closest)

#print("Accuracy of .knn() is: ", acc_score)

