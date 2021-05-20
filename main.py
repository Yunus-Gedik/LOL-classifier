import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("high_diamond_ranked_10min.csv", sep=",")

# Drop game id, which is not related to game, so may damage our model.
data = data.drop('gameId', axis=1)


# Column that we gonna predict.
predict = "blueWins"

# Divide data into train and test splits
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split( X,Y, test_size=0.1 )



linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test, y_test)

print("Linear regression accuracy: ",acc)
# print( "Mean of cross validation score: ", cross_val_score(linear, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


logreg = LogisticRegression()
logreg.fit(x_train, y_train)

acc = logreg.score(x_test, y_test)

print("Logistic regression accuracy: ",acc)
#print( "Mean of cross validation score: ", cross_val_score(logreg, x_test, y_test, scoring='accuracy', cv=10).mean())
print()

bnb = BernoulliNB(binarize=0.0)
bnb.fit(x_train, y_train)

acc = bnb.score(x_test, y_test)

print("Bernoulli naive bayes accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(bnb, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

acc = clf.score(x_test,y_test)

print("Decision tree accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(clf, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


rforest = RandomForestClassifier(n_estimators=100)
rforest = rforest.fit(x_train,y_train)

acc = rforest.score(x_test,y_test)

print("Random Forest accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(rforest, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


neighbours = 125
knn = KNeighborsClassifier(n_neighbors=neighbours)
knn = knn.fit(x_train,y_train)

acc = knn.score(x_test,y_test)

print("KNN accuracy with ",neighbours," :",acc)
print( "Mean of cross validation score: ", cross_val_score(knn, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


sgdc = SGDClassifier(max_iter=1000, tol=0.01)
sgdc = sgdc.fit(x_train, y_train)

acc = sgdc.score(x_test,y_test)

print("SGD accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(sgdc, x_test, y_test, scoring='accuracy', cv=10).mean())
print()

lsvm = svm.SVC(kernel='linear')
lsvm = lsvm.fit(x_test,y_test)

acc = lsvm.score(x_train,y_train)

print("Linear SVM accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(lsvm, x_test, y_test, scoring='accuracy', cv=10).mean())
print()

psvm = svm.SVC(kernel='poly')
psvm = psvm.fit(x_test,y_test)

acc = psvm.score(x_test,y_test)

print("Polynomial SVM accuracy: ",acc)
print( "Mean of cross validation score: ", cross_val_score(psvm, x_test, y_test, scoring='accuracy', cv=10).mean())
print()


"""
predictions = bnb.predict(x_test)

style.use("ggplot")
pyplot.scatter(y_test, predictions)
pyplot.xlabel("Blue Team Win Rate")
pyplot.ylabel("Prediction")
pyplot.show()
"""
