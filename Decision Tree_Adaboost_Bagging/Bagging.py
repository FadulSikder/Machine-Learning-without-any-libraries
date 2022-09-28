import numpy as np
import pandas as pd

def load_data():
    collum_names = ['Height', 'Weight', 'Age', 'Gender']
    df = pd.read_csv("data_2c.csv",skiprows=1, header=None, names=collum_names)
    
    df1 = df.iloc[:90]
    df2 = df.iloc[90:]

    X_train = df1.iloc[:, :-1].values
    Y_train = df1.iloc[:, -1].values.reshape(-1,1)
    X_test = df2.iloc[:, :-1].values
    Y_test = df2.iloc[:, -1].values.reshape(-1,1)

    
    return X_train, Y_train, X_test, Y_test

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, data, current_depth=0):
        
        X, Y = data[:,:-1], data[:,-1]
        samples_number, features_number = np.shape(X)
        # split until set conditions
        if samples_number>=self.min_samples_split and current_depth<=self.max_depth and not self.all_same(Y):
            # find the best split
            best_split = self.get_best_split(data, samples_number, features_number)
            
            if best_split["info_gain"]>0:

                left_subtree = self.build_tree(best_split["data_left"], current_depth+1)

                right_subtree = self.build_tree(best_split["data_right"], current_depth+1)

                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)

        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, data, samples_number, features_number):
        
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(features_number):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                if threshold == possible_thresholds[len(possible_thresholds)-1]:
                    continue 
                # get current split
                data_left, data_right = self.split(data, feature_index, threshold)

                # check if childs are not null
                if len(data_left)>0 and len(data_right)>0:
                    y, left_y, right_y = data[:, -1], data_left[:, -1], data_right[:, -1]
                    # compute information gain
                    current_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if current_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain

        # return best split
        return best_split
    
    def split(self, data, feature_index, threshold):

        data_left = np.array([row for row in data if row[feature_index]<=threshold])
        data_right = np.array([row for row in data if row[feature_index]>threshold])

        return data_left, data_right
    
    def information_gain(self, parent, l_child, r_child):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            probability_cls = len(y[y == cls]) / len(y)
            entropy += -probability_cls * np.log2(probability_cls)

        return entropy
    
    def all_same(self, Y_items):
        return all(x == Y_items[0] for x in Y_items)
        
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y): 
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(data)
    
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

def accuracy(Y, Y_pred):

        accurate = 0

        for i in range(len(Y)):
            if(Y[i]==Y_pred[i]):
                accurate = accurate + 1

        accuracy_rate = accurate/len(Y)

        return(accuracy_rate)

def main():
    
    X_train, Y_train, X_test, Y_test = load_data()
   
    for i in range(9):
        classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=i+1)
        classifier.fit(X_train,Y_train)

        Y_pred_train = classifier.predict(X_train)
        Y_pred_test = classifier.predict(X_test)
          
        accuracy_rate_train = accuracy(Y_train.flatten().tolist(), Y_pred_train)
        print('\033[1m' + 'Accuracy rate for train data at depth',i+1,':',accuracy_rate_train,' '+ '\033[0m')
        accuracy_rate_test = accuracy(Y_test.flatten().tolist(), Y_pred_test)
        print('\033[1m' + 'Accuracy rate for test data at depth',i+1,':',accuracy_rate_test,' '+ '\033[0m')



    return

if __name__ == '__main__':
    main()