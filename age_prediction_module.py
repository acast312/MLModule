import pandas as pd
import numpy as np
from sklearn import svm
from math import floor
import sys

#there are two clear paths we can take. 1: use regression on the raw data, 2: group accoridng to age and use classification
# we will attempt both.

class PredictionClass(object):
    def __init__(self, training_file=str(), validation_file=str()):
        self.training_data = pd.read_csv(training_file)
        self.validation_data = pd.read_csv(validation_file)
        self.prediction = None
        self.classifier = None
        self.age_dict = {1:'young', 2:'medium', 3:'old'}

    def classification(self):
        X_train, y_train, X_test, y_test = self.classiffication_data()
        self.fit(X_train, y_train, True)
        self.prediction = self.predict(X_test)
        print('\nClassification accuracy score: {score}%'.format(score=round(self.validate(self.prediction, y_test))))
        print('Predicted age groups: {group}'.format(group=self.class_to_string(self.prediction)))


    def regression(self):
        X_train, y_train, X_test, y_test = self.regression_data()
        self.fit(X_train, y_train)
        self.prediction = self.predict(X_test)
        class_int = self.age_to_class(self.prediction)
        y_test = self.age_to_class(y_test)
        print('Regression accuracy score: {score}%'.format(score=round(self.validate(class_int, y_test))))
        print('predicted age: {age}'.format(age=[floor(X) for X in self.prediction]))
        #print(self.class_to_string(class_int))

    def classiffication_data(self):
        training = self.training_data
        validation = self.validation_data
        X_train = [self.format_sex(training['sex']), training['length'], training['diameter'], training['whole_weight'], training['shucked_weight'], training['viscera_weight'], training['shell_weight']]
        y_train = training['rings']
        y_train = self.age_to_class(y_train)
        X_validate = [self.format_sex(validation['sex']), validation['length'], validation['diameter'], validation['whole_weight'], validation['shucked_weight'], validation['viscera_weight'],validation['shell_weight']]
        y_validate = validation['rings']
        y_validate = self.age_to_class(y_validate)
        X_train = np.array(X_train).transpose()
        X_validate = np.array(X_validate).transpose()
        return X_train, y_train, X_validate, y_validate

    def regression_data(self):
        training = self.training_data
        validation = self.validation_data
        X_train = [self.format_sex(training['sex']), training['length'],training['diameter'],training['whole_weight'],training['shucked_weight'],training['viscera_weight'],training['shell_weight']]
        y_train = training['rings']
        X_validate = [self.format_sex(validation['sex']), validation['length'],validation['diameter'],validation['whole_weight'],validation['shucked_weight'],validation['viscera_weight'],validation['shell_weight']]
        y_validate = validation['rings']
        X_train = np.array(X_train).transpose()
        X_validate = np.array(X_validate).transpose()
        return X_train, y_train, X_validate, y_validate

    def one_hot(self, array):
        encoded = []
        encoding_dict = {'I': 1, 'F': 2, 'M': 3}
        for value in array:
            encoded.append(encoding_dict[value])
        return encoded

    def fit(self, X, y ,classication = False):
        if classication:
            self.classifier = svm.LinearSVC()
            self.classifier.fit(X,y)
            pass
        else:
            self.classifier = svm.SVR(kernel='linear')
            self.classifier.fit(X,y)
            pass

    def predict(self, X):
        return self.classifier.predict(X)

    def age_to_class(self, array):
        class_array = []
        for value in array:
            if value < 9.0:
                class_array.append(1)
            elif value >= 9.0 and value < 11.0:
                class_array.append(2)
            else:
                class_array.append(3)
        return class_array

    def class_to_string(self, array):
        out_array = []
        for value in array:
            out_array.append(self.age_dict[value])
        return out_array

    def validate(self, prediction, validate):
        itr = 0
        matches = 0
        while itr < len(prediction):
            if prediction[itr] == validate[itr]:
                matches += 1
            itr += 1
        return (matches / len(prediction) * 100)

if __name__ == '__main__':
    prediction = PredictionClass('abbalones_train.txt', 'abbalones_validate.txt')
    prediction.regression()
    prediction.classification()