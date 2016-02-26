import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sknn.mlp import Regressor, Layer
from sklearn.linear_model import perceptron
import matplotlib.pyplot as plt

def getAccuracy(test, predictions):
        correct = 0
        for x in range(len(test)):
                if test[x] == predictions[x]:
                        correct = correct + 1
        percentage =  (correct/float(len(test))) * 100.0
        return percentage

def mainworker(limit1,limit2):
	N=10
	l=[]
	w1=[] # +1 class
	w2=[]#-1 class
	temp=[]
	classlist=[]
	countlist=[]
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
	f.close()
	X=np.array(l)
	y=np.array(classlist)
	w=l

	X=np.array(l)
	y=np.array(classlist)
	karray=[2,3,4,5]
	for k in karray:
		kf=cross_validation.KFold(11054, n_folds=k)
		averager=[]
		for train_index,test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			train_data=X_train
			train_label=y_train
        		test_data=X_test
       			test_label=y_test
        		net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
			net.fit(train_data,train_label)	
			predicted = net.predict(test_data)
			accuracy = net.score(test_data, test_label)*100 
			averager.append(accuracy)
		answer=np.mean(averager)
		print "The accuracy for",k,"th fold is:",answer,
		print '\n' 
	#print "Accuracy   " + str(net.score(train_data,train_label)*100) + "%"


print "Address bar features removed----------"
mainworker(0,11)
print "Abnormal features removed------"
mainworker(12,17)
print "HTML Javascript features removed--------"
mainworker(18,22)
print "Domain based features removed--------"
mainworker(23,29)

	
"""
X=np.array(a)
pca = PCA(n_components=3)
print pca.fit(X)
print pca.transform(X)
print(pca.explained_variance_ratio_)"""
"""xaxis = tuple(x[0] for x in w1)
yaxis = tuple(x[1] for x in w1)
plot.plot(xaxis,yaxis,'bo')
xaxis = tuple(x[0] for x in w2)
yaxis = tuple(x[1] for x in w2)
plot.plot(xaxis,yaxis,'ro')
# x = range(0,(len(a))*100,len(a))
# #x= range(-1*(len(a)+1)/2,(len(a)+1)/2)
# # y=[]
# # for i in x:
# # 	z=-(a[0]*i + a[2])/float([1])
# # 	y.append(z)
# plot.plot(x, a)
plot.xlabel('value of margins')
plot.ylabel('no. of corrections')
plot.show()"""
"""a = 0.01
b=0.01
c=0.01
for i in w:
	print i
#t=[1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
i=0
y=[]
p=[]
while i < len(w):
	p.append(w[i])
  #  p.append(t[i])
	y.append(p)
	p=[]
	i=i+1
o=1
print y
print len(w),'2345654323456543q'
-----------------------------
while 1:
	flag=0
	i=0
	while i < len(w):
	#    print "'",i,"'"
		k=(a*w[i][0])+(b*w[i][1])+(c*w[i][2])
		if k<0:
			flag = 1
		a = a+(w[i][0])
			b = b+(w[i][1])
			c = c+(w[i][2])
	i=i+1
	if flag == 0:
	#print 'yiippeee'
	break

print a, b, c

xaxis = tuple(x[0] for x in w1)
yaxis = tuple(x[1] for x in w1)
plot.plot(xaxis,yaxis,'o')
xaxis = tuple(x[0] for x in w2)
yaxis = tuple(x[1] for x in w2)
x= range(-15,15)
y=[]
for i in x:
	z=-(a*i + c)/float(b)
	y.append(z)
plot.plot(x, y)
plot.plot(xaxis,yaxis,'ro')
plot.xlabel('value of margins')
plot.ylabel('no. of corrections')
plot.show()"""
