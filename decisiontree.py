import numpy as np
from scipy.stats import mode
from sklearn.cross_validation import train_test_split
from scipy.stats import entropy as entropy
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
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
        #if the user wants numerical data, convert it
        if convert_nominal is True:
            self.convert_numerical_data(self.ds)
        else: #otherwise convert to no
            self.convert_nominal_data(self.ds)
            self.standardize_data()
       # print("The dataframe after: \n", self.ds)

        self.data = self.ds.loc[:, : ds_num_col - 2]
        self.targets = self.ds[ds_num_col - 1]

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
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





class DecisionTree:
    def __init__(self, data, targets, inputs):
        self.total_data = data
        self.tree = None
        self.targets = targets
        self.inputs = inputs
        self.classes = np.unique(targets)
        self.nInputs = np.shape(inputs)[0]
        self.closest = np.zeros(self.nInputs)
        self.target_list = np.unique(targets)
        self.attribute_list = []
        for i in range(0,len(self.total_data.columns)):
            self.attribute_list.append(self.Attribute(i))
        print("ATTRIBUTE LIST: ", [attr.name for attr in self.attribute_list])
        print("NUM INPUTS:\n ", self.nInputs)

    class Attribute:
        def __init__(self, name):
            self.name = name
            self.entropy = 0

    def get_entropies(self, data):
        entropies = []
        #print("data in getentrop:", data)
        for i in range(0, len(self.attribute_list)):
            unique_items =  np.unique(data[i])
            #print("The unique items: ", unique_items)
            probs = self.calculate_prob(data[i], unique_items)
            my_entropy = entropy(probs,qk=None, base=2)
            self.attribute_list[i].entropy = my_entropy
        print("ATTRIBUTE LIST AND ENTROPIES:")
        for i in self.attribute_list:
            print(i.name, i.entropy)

    def calculate_prob(self, col, unique_items):
        counts = []
        probs = []
        for i in unique_items:
            probs.append(col.value_counts(1)[i])
            counts.append(col.value_counts(0)[i])
        #print(counts)
        #print(probs)
        return probs

    class AttributeNode:
        def __init__(self, node_col, data_set, is_root_node=False):
            self.node_col = node_col
            self.data_set = data_set
            self.children = []
            self.is_root_node = is_root_node

        def add_child(self, obj):
            self.children.append(obj)

        def print_tree(self, level):
            print(("\t" * level), "ATTRIBUTE: ", self.node_col, "level: ", level, "isroot?", self.is_root_node)
            if self.children:
                print("about to do",  len(self.children), "children: ")
                for child in self.children:
                    child.print_tree(level + 1)

        def isLeaf(self):
            return False

    class LeafNode:
        def __init__(self, catagory):
            self.catagory = catagory

        def print_tree(self, level):
            print(('\t' * level), "Leaf: ", self.catagory)
        def isLeaf(self):
            return True


    class DataPoint:
        def __init__(self, name):
            self.name = name
            self.count = 0
        def increment(self):
            self.count += 1


    def build_tree(self, node, data, targets, attributes_to_split_on, level, data_points=[]):
        #lowest_entropy = max(attribute.entropy for attribute in self.attribute_list)
        if attributes_to_split_on:
            print("STARTING LIST: ", [att.name for att in attributes_to_split_on])
            # gets the index column of the attribute with the highest info gain or lowest entropy
            lowest_entropy_index = min(enumerate(attributes_to_split_on), key=lambda x: x[1].entropy)[0]
            print("lowest entropy index: ", lowest_entropy_index)
            if self.tree is None:
                node = self.tree = self.AttributeNode(lowest_entropy_index, data, True)
            else:
                node.add_child(self.AttributeNode(lowest_entropy_index, data))

            possible_values_in_col = np.unique(data[lowest_entropy_index])
            print("UNIQUE DATA: ", possible_values_in_col)
            for i in possible_values_in_col:
                new_set = data[data[lowest_entropy_index] == i]
                # this new set contains all of the times attribute i appeared (0's, 1's, 2's...) in colum:lowest_entropy_index
                print("new set of: ", i, "\n", new_set)
                set_indicies = new_set.index.values
                # i use the DataPoint class to track the occurance number of each target
                data_points = []
                for target in self.target_list:
                    data_points.append(self.DataPoint(target))
                    # count occurences of each target in the new_set
                for index in set_indicies:
                    for j in data_points:
                        if j.name is targets[index]:
                            j.increment()
                # pure tracker holds the number of targets that appear in a subset
                pure_tracker = 0
                for data_point in data_points:
                    if data_point.count > 0:
                        # there is a possibility this could be pure, save the name of the target for a potential leaf
                        leaf_name = data_point.name
                        pure_tracker += 1
                    print("count for: ", data_point.name, ": ", data_point.count)
                    # when the subset is not pure (there are more than 1 targets in it)
                if pure_tracker > 1:
                    # make attribute node
                    copy_list =  list(attributes_to_split_on)
                    print("deleting lowest entropy index: ", lowest_entropy_index, "from: ", [item.name for item in copy_list])
                    copy_list.remove(copy_list[lowest_entropy_index])
                    print("about to go up a level from: ", level)
                    self.build_tree(node, new_set, targets, copy_list, level + 1, data_points)
                    # otherwise we create a leaf node at this spot
                else:
                    print("adding leaf: ", leaf_name)
                    node.add_child(self.LeafNode(leaf_name))
        else:
            print("the data is all the same for diff targets")
            name = ''
            highest = 0
            for point in data_points:
                if point.count > highest:
                    name = point.name
            node.add_child(self.LeafNode(name))





print("**********************START**************************")

my_dataset = Dataset('iris.csv', .3, 3333, True)

decision_tree = DecisionTree(my_dataset.data_train, my_dataset.target_train, my_dataset.data_test)

decision_tree.get_entropies(decision_tree.total_data)
print("FULL SET: ", my_dataset.data_train, my_dataset.target_train)
decision_tree.build_tree(decision_tree.tree, decision_tree.total_data, decision_tree.targets, decision_tree.attribute_list, 1)

decision_tree.tree.print_tree(1)

#closest = nearestneighbor.knn()

#closest = closest.astype(int)

#print("closest: ", closest)

#acc_score = accuracy_score(my_dataset.target_test, closest)

#print("Accuracy of .knn() is: ", acc_score)

