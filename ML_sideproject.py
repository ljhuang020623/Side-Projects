## 1. Load The Data
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
...

#Load Dataset
url = "Dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)


## 2. Summarize the Dataset(pandas)

#Dimension of Dataset
#print(dataset.shape)

#Peek at the Data 
#print(dataset.head(n)), n means first n rows of the data
#print(dataset.head(20))

#discription
#statistical summary
#print(dataset.describe())

#class distribution
#print(dataset.groupby('class').size())

## 3. Data Visualization(matplotlib)

#box and whisker plots
#dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
#plt.show()

#histogram
#dataset.hist()
#plt.show()

#scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

## 4. Evaluate Some Algorithms

#Split-out validation dataset(numpy)
array = dataset.values
x = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size = 0.2, random_state = 1)

#Build model
#1. Logistic regression(LR)
#2. Linear Discriminant Analysis(LDA)
#3. K-Nearest Neighbors(KNN)
#4. Classification and Regression Trees(CART)
#5. Gaussan Naive Bayes(NB)
#6. Supporrt Vector Machines(SVM)

...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
...
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


