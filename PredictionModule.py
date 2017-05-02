import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import svm



class Regression():
    def __init__(self, X, y):
        self.prediction = None
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = None
        self.xencoder = LabelEncoder()
        self.yencoder = LabelEncoder()
        self.yencoded = False
        self.data_encoded = False
        self.predictor = None

    def encode_strings(self, array=None):
        if array == None:
            if len(self.X) > len(self.X.transpose()):
                self.X = self.X.transpose()
            iterv = 0
            while iterv < len(self.X):
                if isinstance(self.X[iterv][0], str):
                   self.xencoder.fit(self.X[iterv])
                   self.X[iterv] = self.xencoder.transform(self.X[iterv])
                iterv += 1
            self.X = self.X.transpose()
            if isinstance(self.y, str):
                self.yencoder.fit(self.y)
                self.y = self.yencoder.transform(self.y)
                self.yencoded = True
            self.data_encoded = True
        else:
            if len(array) > len(array.transpose()):
                array = array.transpose()
            iterv = 0
            while iterv < len(array):
                if isinstance(array[iterv][0], str):
                   array[iterv] = self.xencoder.transform(array[iterv])
                iterv += 1
            return array.transpose()

    def decode(self, array):
         return self.yencoder.inverse_transform(array)

    def fit_model(self, kernel='linear'):
        self.encode_strings() if not self.data_encoded else None
        self.predictor = svm.SVR(kernel=kernel)
        self.predictor.fit(self.X[len(self.X) - 100:], self.y[len(self.y) - 100:])

    def predict(self, X_eval):
        X_eval = np.array(X_eval)
        X_eval = self.encode_strings(array=X_eval)
        self.prediction = self.predictor.predict(X_eval)
        if self.yencoded:
            return self.decode(self.prediction)
        else:
            return self.prediction

    def validate_model(self, x_val=[], y_val=[], errange=2):
        if len(x_val) == 0 or len(y_val) == 0:
            x_val, y_val = self.X, self.y
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        p = self.predict(x_val)

        match = 0
        for i in range(len(y_val)):
            if (p[i] >= y_val[i] - errange) and (p[i] <= y_val[i] + errange):
                match += 1

        return(match/len(y_val))*100



class Classification():
    def __init__(self, X, y, classrange=3):
        self.prediction = None
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = None
        self.xencoder = LabelEncoder()
        self.yencoder = LabelEncoder()
        self.yencoded = False
        self.data_encoded = False
        self.predictor = None
        self.classrange = classrange

    def encode_strings(self, array=None):
        if array == None:
            if len(self.X) > len(self.X.transpose()):
                self.X = self.X.transpose()
            iterv = 0
            while iterv < len(self.X):
                if isinstance(self.X[iterv][0], str):
                   self.xencoder.fit(self.X[iterv])
                   self.X[iterv] = self.xencoder.transform(self.X[iterv])
                iterv += 1
            self.X = self.X.transpose()
            if isinstance(self.y, str):
                self.yencoder.fit(self.y)
                self.y = self.yencoder.transform(self.y)
                self.yencoded = True
            self.data_encoded = True
        else:
            if len(array) > len(array.transpose()):
                array = array.transpose()
            iterv = 0
            while iterv < len(array):
                if isinstance(array[iterv][0], str):
                   array[iterv] = self.xencoder.transform(array[iterv])
                iterv += 1
            return array.transpose()

    def decode(self, array):
         return self.yencoder.inverse_transform(array)

    def fit_model(self):
        self.encode_strings() if not self.data_encoded else None
        self.predictor = svm.LinearSVC()
        fit_y = self.create_classes(fit=True)
        self.predictor.fit(self.X[len(self.X) - 100:], fit_y[len(fit_y) - 100:])

    def predict(self, X_eval):
        X_eval = np.array(X_eval)
        X_eval = self.encode_strings(array=X_eval)
        self.prediction = self.predictor.predict(X_eval)
        if self.yencoded:
            return self.decode(self.prediction)
        else:
            return self.prediction

    def validate_model(self, x_val=[], y_val=[]):
        if len(x_val) == 0 or len(y_val) == 0:
            x_val, y_val = self.X, self.y
        x_val = np.array(x_val)
        y_val = self.create_classes(y_val)
        y_val = np.array(y_val)
        p = self.predict(x_val)

        match = 0
        for i in range(len(y_val)):
            if (p[i] >= y_val[i]) and (p[i] <= y_val[i]):
                match += 1

        return(match/len(y_val))*100


    def create_classes(self, y=None, fit=False):
        converted = []
        if fit:
            tmp_ = []
            tmp_.extend(self.y)
            tmp_.sort()
            values = list(set(tmp_))
            self.conv_array = np.array_split(values, self.classrange)
            for value in self.y:
                for split in range(self.classrange):
                    min_ = self.conv_array[split][0]
                    max_ = self.conv_array[split][len(self.conv_array[split]) - 1]
                    if value >= min_ and value < max_:
                        converted.append(split)
        else:
            for value in y:
                for split in range(self.classrange):
                    min_ = self.conv_array[split][0]
                    max_ = self.conv_array[split][len(self.conv_array[split]) - 1]
                    if value >= min_ and value < max_:
                        converted.append(split)
        return converted


#the following code is intended as an 'example'

if __name__ == '__main__':
    import pandas as pd
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train = pd.read_csv('abbalones_train.txt')
        X = train[['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']]
        y = train['rings']
        X_train = X[:len(X) - 500]
        y_train = y[:len(y) - 500]
        X_val = X[len(X)-500:]
        y_val = y[len(y) - 500:]
        print('testing Regression class')
        rlf = Regression(X_train, y_train)
        print('fit and validate model')
        rlf.fit_model()
        print(rlf.validate_model(X_val,y_val,errange=4), '%\n')
        print('Now let\'s test the Classification class')
        clf = Classification(X_train, y_train, classrange=5)
        print('fit and validate model')
        clf.fit_model()
        print(clf.validate_model(X_val,y_val), '%')
