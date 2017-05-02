### Machine Learning made easy

The goal of this library is to provide non-Data Scientist with access to some Machine Learning
tools.

Currently the library has 2 main classes (but is being extended):
1) Regression/prediction capabilities
2) Classification capabilities.

Use of the Library can be seen inside the source code.

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

