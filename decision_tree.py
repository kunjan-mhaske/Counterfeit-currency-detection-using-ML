__author__ = 'KSM'

"""
Author: Kunjan Mhaske

This program generates the decision tree based on the entropy and gain values of given dataset with the class
labels of 0 and 1 to predict the authentication of bank note.
Here 0 means Fake Note and 1 means Genuine Note
"""

import numpy as np
import math, random

R = 5 # Reruns for cross validations

class TreeNode:
    """
    This class maintains the tree node which contains the attribute information and tree structure
    """
    __slots__ = 'attribute','splitpoint','gainvalue','left_side','right_side','prediction'
    def __init__(self, attribute, splitpoint, gain, left=None, right=None, value=None):
        """
        Init method to instantiate the tree node
        :param attribute: attribute of particular node ( useful for leaf nodes )
        :param splitpoint: splitpoint
        :param gain: gain of partuiular substet
        :param left: left branch of tree
        :param right: right branch of tree
        :param value: prediction value ( 0 or 1) ( useful for leaf node )
        """
        self.attribute = attribute
        self.splitpoint = splitpoint
        self.gainvalue = gain
        self.left_side = left
        self.right_side = right
        self.prediction = value

    def __str__(self):
        """
        String representation of tree node
        :return: string
        """
        return str(self.attribute)+":"+str(self.splitpoint)

class DecisionTree:
    """
    This class handles the tree nodes and perform various processes on it
    """
    __slots__ = 'attributes','classes', 'totalEntropy', 'totalPCount', 'totalNCount'
    def __init__(self, attribs, classes):
        """
        Initailize the decision tree object
        :param attribs: attribute counts in the dataset
        :param classes: class counts in the dataset
        """
        self.attributes = attribs
        self.classes = classes
        self.totalEntropy = 0
        self.totalPCount = 0
        self.totalNCount = 0

    def calTotalEntropy(self, data):
        """
        It calculates total entropy of given data
        :param data: data array
        :return: entropy, class1 counts, class0 counts
        """
        nCount = 0
        for obs in data:
            if obs[-1] == 0.0 :
                nCount += 1
        pCount = len(data) - nCount
        totalEntropy = self.calEntropy( pCount/(pCount+nCount) )
        return totalEntropy, pCount, nCount

    def calEntropy(self,value):
        """
        It calculates the entropy of particular value/fraction
        :param value: value/ fraction of which the entropy is to be calculated
        :return: entropy
        """
        if (value == 0) or (value==1):
            return 0
        else:
            return - ( value*math.log2(value) + (1-value)*math.log2(1-value))

    def calRem(self, data):
        """
        It calculates the Rem value required in gain calculation
        :param data: data array
        :return: rem value, subset class1 count, subset class0 count
        """
        knCount = 0
        for obs in data:
            if obs[-1] == 0.0 :
                knCount += 1
        kpCount = len(data) - knCount
        if (kpCount+knCount) == 0:
            return 0, kpCount, knCount
        rem = ( (kpCount+knCount)/(self.totalPCount+self.totalNCount) ) * self.calEntropy( kpCount/(kpCount+knCount) )
        return rem, kpCount, knCount

    def calPredict(self,data):
        """
        It gives subset class prediction of class 1 or 0
        :param data: subset data array
        :return: 0 or 1 prediction
        """
        nCount = 0
        for obs in data:
            if obs[-1] == 0.0 :
                nCount += 1
        pCount = len(data) - nCount
        if nCount <= pCount:
            return 1.0
        else:
            return 0.0

    def splitDataset(self,point,attrib,data):
        """
        This method use to split the dataset into left and right part based on the attribute and splitpoint
        :param point: splitpoint
        :param attrib: attribute
        :param data: data array
        :return: left and right part of data
        """
        left = []
        right = []
        if point == None:
            return left, right
        for obs in data:
            if obs[attrib] <= point:
                left.append(list(obs))
            else:
                right.append(list(obs))
        return left, right

    def findSplits(self,data, altAttrib):
        """
        This method is used to find maximum profitable splitpoint based on gain calculation of particular subset
        :param data: data array
        :param altAttrib: already considered attributes
        :return: dictionary of attributes with gain and splitpoint values
        """
        attribDict = {}
        # print("already considered arttribs:",altAttrib)
        for a in range(self.attributes):
            if a in altAttrib:
                continue
            # print("considering attrib:",a)
            attribDict[a] = {}
            subset = []
            for obs in data:
                subset.append(obs[a])
            subset.sort()
            maxgain = -9999
            splitPoint = 0
            for i in range(len(subset)-1):
                point1 = subset[i]
                point2 = subset[i+1]
                avg = (point1+point2)/2
                left, right = self.splitDataset(avg,a,data)
                leftRem, _, _ = self.calRem(left)
                rightRem, _, _ = self.calRem(right)
                gain = self.totalEntropy - (leftRem+rightRem)
                if gain > maxgain:
                    maxgain = gain
                    splitPoint = avg
            attribDict[a]['gain'] = maxgain
            attribDict[a]['splitpoint'] = splitPoint
        return attribDict

    def getMaxGains(self,attribDict):
        """
        This method gives the attribute having maximum gain in subset
        :param attribDict: attribute dictionary with gain and splitpoint
        :return: attribute, maximum gain
        """
        # print("Getting Max gain:")
        # for key in attribDict:
        #     print(key,":",attribDict[key])
        gainarr = []
        attribarr = []
        for a in attribDict:
            gain = attribDict[a]['gain']
            attribarr.append(a)
            gainarr.append(gain)
        if gainarr == []:
            maxgain = None
            attrib = None
        else:
            maxgain = max(gainarr)
            atindex = gainarr.index(maxgain)
            attrib = attribarr[atindex]
        # print('attrib: ',attrib,' and maxgain:',maxgain)
        return attrib, maxgain

    def recursion(self, data, altAttrib):
        """
        This recursion is used to generate the decision trees based on the nodes which contains the prediction
        and the tree is built based on the splitpoints
        :param data: data array
        :param altAttrib: already considered attributes list
        :return: Tree root
        """
        if (len(altAttrib)== self.attributes) or (data == []):
            return
        # find split points
        attribDict = self.findSplits(data,altAttrib.copy())
        # get attrib of max gain
        attrib, maxgain = self.getMaxGains(attribDict)

        if (attrib != None) or (maxgain != None) :
            splitPoint = attribDict[attrib]['splitpoint']
        else:
            splitPoint = None
            attrib = None
            maxgain = None

        # split dataset based on splitpoint
        left, right = self.splitDataset(splitPoint,attrib,data)
        left = np.asarray(left)
        right = np.asarray(right)

        # already considered attributes
        altAttrib.append(attrib)
        # TreeNode(self, attribute, splitpoint, gain, left=None, right=None, count=0)
        return TreeNode(attrib, splitPoint, maxgain, self.recursion(left,altAttrib.copy()), self.recursion(right,altAttrib.copy()), self.calPredict(data) )

    def traverseTree(self,root):
        """
        This method used to traverse the tree
        :param root: root of tree
        :return: none
        """
        if root:
            print("left of",root.attribute)
            self.traverseTree(root.left_side)
            print('root:',root.attribute)
            print("right of",root.attribute)
            self.traverseTree(root.right_side)

    def training(self,data):
        """
        This method takes the data and builds the decision tree from it
        :param data: data array
        :return: root of decision tree
        """
        print("Training Start......")
        # calculate total entropy
        self.totalEntropy, self.totalPCount, self.totalNCount = self.calTotalEntropy(data)
        # attrib list
        altAttrib = []
        parseData = data.copy()
        root = self.recursion(parseData, altAttrib)
        print("Decision Tree Built.")
        # self.traverseTree(root)
        print("Training Done.")
        print()
        return root

    def __testingHelper(self,obs,root):
        """
        This is recursion helper function to traverse the decision tree and return the prediction
        :param obs: instance of data
        :param root: root of tree
        :return: prediction value
        """
        pred = None
        if root == None:
            return pred
        if root.left_side is None and root.right_side is None:
            return root.prediction
        attrib = root.attribute
        split = root.splitpoint
        if obs[attrib] <= split:
            pred = self.__testingHelper(obs, root.left_side)
        elif obs[attrib] > split:
            pred = self.__testingHelper(obs, root.right_side)
        return pred

    def testing(self,root,data):
        """
        This method is used to make prediciton on test data based on decision tree
        :param root: root of decision tree
        :param data: test data
        :return: accuracy of prediction
        """
        print("Testing...")
        correctCount = 0
        wrong = 0
        for obs in data:
            # print(obs, end='-- ')
            result = self.__testingHelper(obs,root)
            if result == obs[-1]:
                # print(result)
                correctCount += 1
            else:
                wrong += 1
                # print(result,"WRONG PREDICTION")
        print("Total Testing Data Count:",len(data))
        print("CorrectCount:",correctCount)
        accuracy = 100*(correctCount / len(data))
        print("Accuracy: %f percent"%accuracy)
        print("__________________________________________________________")
        return accuracy

    def crossValidation(self,data):
        """
        This method cross validates the entire dataset for the decision tree
        :param data: data array
        :return: accuracy array of cross validations
        """
        # for particular range
        print("Cross Validation for ratio 20:80 = training : testing data")
        print()
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
            test_data = np.concatenate((test_1,test_2), axis=0)
            root = self.training(train_data)
            acc = self.testing(root,test_data)
            accuracy.append(acc)
            print()
        return accuracy

    def majorityTraining(self,data):
        """
        This method gives label based on majority class contained in the dataset
        :param data: data array
        :return: label of majority class
        """
        print("Training Start......")
        self.totalEntropy, self.totalPCount, self.totalNCount = self.calTotalEntropy(data)
        print("Total Points in class 0:",self.totalNCount)
        print("Total Points in class 1:",self.totalPCount)
        if self.totalPCount >= self.totalNCount:
            print("Majority Class:",1)
            return 1
        else:
            print("Majority Class:", 0)
            return 0

    def majorityTesting(self,outcome,data):
        """
        This method gives prediction of class based on the outcome value it takes
        :param outcome: prediction value
        :param data: test data
        :return: none
        """
        correctCount = 0
        for obs in data:
            if int(obs[-1]) == outcome:
                correctCount += 1
        print("Total Testing Data Count:",len(data))
        print("CorrectCount:",correctCount)
        accuracy = 100*(correctCount / len(data))
        print("Accuracy: %f percent"%accuracy)
        print("__________________________________________________________")

    def randomTrainingAndTesting(self,data):
        """
        This method gives prediction based on random outcomes of classes of data
        :param data: data
        :return: none
        """
        correctCount = 0
        positiveCount = 0
        for obs in data:
            pred = random.randint(0,1)
            if int(obs[-1]) == pred:
                correctCount += 1
                if pred == 1:
                    positiveCount += 1
        self.totalEntropy, self.totalPCount, self.totalNCount = self.calTotalEntropy(data)
        print("Original Class 0 count:",self.totalNCount)
        print("Predicted Class 0 count:",len(data)-positiveCount)
        print()
        print("Original Class 1 count:", self.totalPCount)
        print("Predicted Class 1 count:",positiveCount)
        print()
        print("Total Testing Data Count:", len(data))
        print("CorrectCount:", correctCount)
        accuracy = 100 * (correctCount / len(data))
        print("Accuracy: %f percent" % accuracy)
        print("__________________________________________________________")


def load_data(filename):
    """
    This method prettify the data to parse into classifier
    :param filename: name of dataset file
    :return: array of data
    """
    data = np.array([[float(x) for x in line.strip().split(',')] for line in open(filename).readlines()])
    print('Loaded %d observations.'%len(data))
    return data

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
    return variables, classesCount, classes

if __name__ == '__main__':
    data = load_data('banknote.txt')
    # data = load_data('test.txt')
    attribs , classes, classList = autoDetect(data)
    print("Attributes Count:",attribs," and Classes:", classList)
    print("-------------------: Predictions Based on Majority Class :------------------")
    major = DecisionTree(attribs, classes)
    prediction = major.majorityTraining(data.copy())
    major.majorityTesting(prediction,data.copy())
    print()
    print("-------------------: Predictions Based on Random Outcome :------------------")
    randOut = DecisionTree(attribs,classes)
    randOut.randomTrainingAndTesting(data.copy())
    print()
    print("-------------------: Predictions Based on Decision Tree :------------------")
    # can use to re run the validation for multiple times.
    print("Total",R,"Reruns...")
    accuracy = []
    for x in range(R):
        decisionTree = DecisionTree(attribs, classes)
        random.shuffle(data)
        accuracy.append(decisionTree.crossValidation(data))
    print("Accuracies: ", accuracy)
    print("Average accuracy:", np.average(accuracy))
    print()

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Prediction for random single point in dataset:")
    decTreeObj = DecisionTree(attribs, classes)
    sample = data.copy()
    rand = random.randint(0, len(sample) - 1)
    test_data = [sample[rand]]
    print("Testing Data:",test_data)
    np.delete(sample, rand)
    train_data = sample
    root = decTreeObj.training(train_data)
    decTreeObj.testing(root, test_data)
    print()

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Parsing whole data to Training and Testing :")
    decTreeObj1 = DecisionTree(attribs, classes)
    sample2 = data.copy()
    random.shuffle(sample2)
    root = decTreeObj1.training(sample2.copy())
    decTreeObj1.testing(root,sample2.copy())