from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn import svm
from sklearn import cross_validation
import numpy as np
from sknn.mlp import Regressor, Layer

def getAccuracy(test, predictions):
	correct = 0
	for x in range(len(test)):
		if test[x] == predictions[x]:
			correct = correct + 1
	percentage =  (correct/float(len(test))) * 100.0
	return percentage



def splitList(someList, testing, list1, list2):
	counter = 0
	for something in someList:
		if(counter < testing ):
			list2.append(something)
			counter = counter + 1
		else:
			list1.append(something)

def mainworker(limit1,limit2):
	N=10
	l=[]
	w1=[] # +1 class
	w2=[]#-1 class
	temp=[]
	classlist=[]
	f=open("pdata.txt")
	for line in f:
        	x=(line.strip("\n")).split(",")
        	temp=[]
        	for i in xrange(len(x)):
			x[i]=int(x[i])
			temp.append(x[i])
        	clas=temp.pop()
		temp=temp[:limit1]+temp[limit2+1:]
        	l.append(temp)
       		classlist.append(clas)
       		"""if(temp[-1]==-1):
                	w2.append(temp)
       		else:
                	w1.append(temp)"""
	f.close()
	X=np.array(l)
	y=np.array(classlist)

	X=np.array(l)
	y=np.array(classlist)
	karray=[2,3,4,5]

	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis"]

	classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			AdaBoostClassifier(), GaussianNB(), LinearDiscriminantAnalysis()]

	for name, clf in zip(names, classifiers):
		print name,"------------------"
		for k in karray:
			kf = cross_validation.KFold(11054, n_folds=k)
			averager=[]
			for train_index,test_index in kf:
			#print("TRAIN:", train_index, "TEST:", test_index)
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
			#print X_train, len(X_test), len(y_train), len(y_test)
				train_data=[]
				test_data=[]
				train_label=[]
				test_label=[]
				X1 = X_train#train_data
				Y1 = y_train#train_label	
				#clf = PassiveAggressiveClassifier()
				#clf = svm.SVC(kernel='linear')
				clf.fit(X1,Y1)
				score = clf.score(X_test, y_test)
				Z = X_test#test_data
				predicted = clf.predict(Z)
				accuracy = getAccuracy(predicted, y_test)#test_label)
				averager.append(accuracy)
			answer=np.mean(averager)
			print "The mean for",k,"fold is:"
			print answer
			
print "Address bar features removed----------"
mainworker(0,11)
print "Abnormal features removed------"
mainworker(12,17)
print "HTML Javascript features removed--------"
mainworker(18,22)
print "Domain based features removed--------"
mainworker(23,29)

