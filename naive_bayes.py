__author__ = 'KSM'

"""
Authors:  alp@cs.rit.edu and kunjan mhaske

This program is implementation of naive bayes classifier which gives predictions as well as accuracy of prediction

    Some potential improvements:
        o automatically determine the number of observable variables (N) and classes (Y).
        o add cross-validation
        o make parameter learning more efficient : took median

"""

import numpy as np
from scipy.stats import norm
import math, random, warnings
warnings.filterwarnings("ignore")

R = 5 # Rerun times for cross validation

def autoDetect(data):
    """
    This method detects the classes and attribute counts from the data
    :param data: data
    :return: attribute count and class count
    """
    classes = []
    variables = len(data[0]) - 1
    for obs in data:
        if obs[-1] not in classes:
            classes.append(obs[-1])
    classesCount = len(classes)
    return variables, classesCount

def load_data(filename):
    """
    This method prettify the data to parse into classifier
    :param filename: name of dataset file
    :return: array of data
    """
    data = np.array([[float(x) for x in line.strip().split(',')] for line in open(filename).readlines()])
    print('Loaded %d observations.'%len(data))
    return data

def condprob(x, n, y, params):
    """
    This method calculated the conditional probability based on probability density function
    :param x: value
    :param n: attribute
    :param y: class
    :param params: learned values dictionary
    :return: conditional probability
    """
    return norm.pdf(x, params[n][y]['median'], params[n][y]['var'])

def learn(data):
    """
    This method learns the information and train the naive bayes classifier
    :param data: input data
    :return: parameter dicitionary
    """
    params = {}
    for n in range(N):
        params[n] = {}
        for y in range(Y):
            params[n][y] = {}
            subset = []
            for obs in data:
                if obs[-1] == y:
                    subset.append(obs[n])
            params[n][y]['median'] = np.median(subset)
            params[n][y]['var'] = np.var(subset)
    return params

def classify(obs, params):
    """
    This method gives the predictions based on the parameters and conditional probabilities
    :param obs: instance of data
    :param params: parameter dictionary
    :return: list of probabilities outcome
    """
    ans = []
    for y in range(Y):
        prob  = 1
        for n in range(N):
            prob *= condprob(obs[n], n, y, params)
        ans.append(prob)
    return ans

def crossValidation(data):
    """
    This method cross validates the entire dataset for the naive bayes classifier
    :param data: data array
    :return: accuracy array of cross validations
    """
    # for particular range
    archPoints = [0,20,40,60,80,100]
    accuracy = []
    for i in range(len(archPoints)-1):
        splitPoint1 = math.floor((archPoints[i]/100) * len(data))
        splitPoint2 = math.floor((archPoints[i+1]/100) * len(data))
        print("Traning data from",splitPoint1,"to",splitPoint2-1)
        train_data = data[splitPoint1:splitPoint2]
        print("Testing data from",0,"to",splitPoint1,"and from",splitPoint2-1,"to",len(data)-1)
        test_1 = data[:splitPoint1]
        test_2 = data[splitPoint2:]
        if test_1 == []:
            test_data = test_2
        elif test_2 == []:
            test_data = test_1
        else:
            test_data = np.concatenate((test_1,test_2), axis=0)
        params = learn(train_data)              # training
        correct = 0
        for obs in test_data:                   # testing
            result = classify(obs, params)
            result = np.array(result) / np.sum(result)
            if np.argmax(result) == obs[-1]:
                correct += 1
        accuracy.append(100 * (correct / len(test_data)))
        print("Accuracy:",100 * (correct / len(test_data)))
        print()
    return accuracy

def demo():
    """
    This is driver function for all the processes in naive bayes classifier
    :return: None
    """
    file = input("Enter dataset file name:")
    data = load_data(file)
    attributes, classes = autoDetect(data)
    global N
    N = attributes
    global Y
    Y = classes
    print("Attributes Count:",N," and Classes Count:", Y)
    print("-------------------: Cross Validation of Naive Bayesian classifier :------------------")
    print("Ratio 20:80 = training : testing data")
    print("Running for",R,"times")
    print()
    accuracy = []
    sample = data.copy()
    for x in range(R):
        random.shuffle(sample)
        accuracy.append(crossValidation(sample))
    print("Cross Validation Done.......")
    print("Accuracies: ", accuracy)
    print("Average accuracy in cross validation:", np.average(accuracy))
    print()
    print("-------------------: Predictions Based on Naive Bayesian classifier  :------------------")
    print("Parsing whole data to Training and Testing")
    sample2 = data.copy()
    params = learn(sample2)              # training
    correct = 0
    for obs in sample2:  # testing
        print(obs, end='-- ')
        result = classify(obs, params)
        result = np.array(result) / np.sum(result)
        if np.argmax(result) == obs[-1]:
            correct += 1
        print(result, end=' -- ')
        print(np.argmax(result))
    accu = (100 * (correct / len(sample2)))
    print("Accuracy ",accu)

if __name__ == '__main__':
    demo()