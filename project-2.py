import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as prepro
from mpl_toolkits.mplot3d import Axes3D
from cvxopt import matrix, solvers
from numpy import array, dot
from qpsolvers import solve_qp
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def generate_data():

	mean_class_pos = [1, 1]
	mean_class_neg = [-1, -1]

	#print(mean_class_pos.shape)
	#print(mean_class_neg.shape)

	covariance_pos = [[3, 2], [2, 3]]
	covariance_neg = [[2, -1], [-1, 2]]

	#print(covariance_pos.shape)
	#print(covariance_neg.shape)

	X_pos, y_pos = np.random.multivariate_normal(mean_class_pos, covariance_pos, 200).T

	#X_pos = np.transpose(np.asmatrix(X_pos))
	#y_pos = np.transpose(np.asmatrix(y_pos))
	#print(X_pos.shape)
	#print(y_pos)

	X_neg, y_neg = np.random.multivariate_normal(mean_class_neg, covariance_neg, 200).T

	#X_neg = np.transpose(np.asmatrix(X_neg))
	#y_neg = np.transpose(np.asmatrix(y_neg))


	X_pos_train = X_pos[:100]
	y_pos_train = y_pos[:100]
	X_pos_test = X_pos[100:200]
	y_pos_test = y_pos[100:200]
	y_pos_label = np.ones(200)

	X_y_pos_train = np.column_stack((X_pos_train, y_pos_train))
	X_y_pos_train = np.column_stack((X_y_pos_train, y_pos_label[:100]))

	X_y_pos_test = np.column_stack((X_pos_test, y_pos_test))
	X_y_pos_test = np.column_stack((X_y_pos_test, y_pos_label[100:200]))

	#print(np.mean(X_y_pos_train))
	


	X_neg_train = X_neg[:100]
	y_neg_train = y_neg[:100]
	X_neg_test = X_neg[100:200]
	y_neg_test = y_neg[100:200]
	y_neg_label = -1*np.ones(200)


	X_y_neg_train = np.column_stack((X_neg_train, y_neg_train))
	X_y_neg_train = np.column_stack((X_y_neg_train, y_neg_label[:100]))

	X_y_neg_test = np.column_stack((X_neg_test, y_neg_test))
	X_y_neg_test = np.column_stack((X_y_neg_test, y_neg_label[100:200]))

	#print(np.mean(X_y_neg_train))


	"""
	fig = plt.figure()
	plots = fig.add_subplot(111)

	plots.scatter(X_y_pos_train[:,0], X_y_pos_train[:,1], c='g', marker = 's', label = 'positive', alpha = 0.5)
	plots.scatter(X_y_neg_train[:,0], X_y_neg_train[:,1], c='r', marker = 'o', label = 'negative', alpha = 0.5)
	plt.xlabel("x")
	plt.ylabel('y')
	plt.show()


	fig = plt.figure()
	plots = fig.add_subplot(111)

	plots.scatter(X_y_pos_test[:,0], X_y_pos_test[:,1], c='g', marker = 's', label = 'positive', alpha = 0.5)
	plots.scatter(X_y_neg_test[:,0], X_y_neg_test[:,1], c='r', marker = 'o', label = 'negative', alpha = 0.5)
	plt.xlabel("x")
	plt.ylabel('y')
	plt.show()
	"""



	data_train = np.concatenate((X_y_pos_train, X_y_neg_train))
	data_test = np.concatenate((X_y_pos_test, X_y_neg_test))


	#print(data_train.shape)
	#print(data_test.shape)

	"""
	fig = plt.figure()
	plots = fig.add_subplot(111)

	plots.scatter(data_train[:,0], data_train[:,1], c='g', marker = 's', label = 'train', alpha = 0.5)
	plots.scatter(data_test[:,0], data_test[:,1], c='r', marker = 'o', label = 'test', alpha = 0.5)
	plt.xlabel("x")
	plt.ylabel('y')
	plt.show()
	"""


	return data_train, data_test

def lda_analysis_library(data_train, data_test):

	X = np.column_stack((data_train[:, 0], data_train[:, 1]))
	y = data_train[:, 2]

	clf = LinearDiscriminantAnalysis()
	clf.fit(X, y)

	X_test = np.column_stack((data_test[:, 0], data_test[:, 1]))

	pred_label = clf.predict(X_test)
	#print(test_label)

	true_label = data_test[:, 2]

	print(accuracy_score(true_label, pred_label))


def lda_analysis(data_train, data_test):

	X = np.column_stack((data_train[:, 0], data_train[:, 1]))
	X_test = np.column_stack((data_test[:, 0], data_test[:, 1]))



	est_mean = np.mean(X)
	print('est_mean = ')
	print(est_mean)


	mean_class_pos = [np.mean(X[0:100, 0]), np.mean(X[0:100, 1])]
	mean_class_pos = np.asmatrix(np.transpose(mean_class_pos))
	print('Mean class positive')
	print(mean_class_pos)

	mean_class_neg = [np.mean(X[100:200, 0]), np.mean(X[100:200, 1])]
	mean_class_neg = np.asmatrix(np.transpose(mean_class_neg))
	print('Mean class negative')
	print(mean_class_neg)

	#print(est_mean)

	X = X - est_mean
	#print(np.transpose(X))
	est_covariance = np.matmul(np.transpose(X), X)/(200-1)

	print('est covariance = ')
	print(est_covariance)

	SIGMA_inv = np.asmatrix(np.linalg.inv(est_covariance))

	print('SIGMA inv = ')
	print(SIGMA_inv)

	w = np.matmul((mean_class_pos - mean_class_neg), SIGMA_inv)

	print('Learnt weights, w = ')
	print(np.asarray(w).flatten())

	#print(mean_class_pos.shape)
	#print(SIGMA_inv.shape)

	b = 0.5 * ( mean_class_pos*SIGMA_inv*np.transpose(mean_class_pos) - mean_class_neg*SIGMA_inv*np.transpose(mean_class_neg))

	print('Learnt bias, b = ')
	print(b) 

	func_of_x = w*np.asmatrix(np.transpose(X_test)) + np.asarray(b).flatten()
	func_of_x = np.transpose(func_of_x)
	#print(func_of_x)

	pred_label = np.where(func_of_x > 0, 1.0, func_of_x)
	pred_label = np.where(pred_label <= 0, -1.0, pred_label)

	#print(test_label)

	true_label = data_test[:, 2]

	print(accuracy_score(true_label, pred_label))
	plot_after_classification(data_test, pred_label)

def qda_using_library(data_train, data_test):
	X = np.asarray(np.column_stack((data_train[:, 0], data_train[:, 1])))
	y = np.asarray(data_train[:, 2])

	#print(X)
	#print(y)
	X_test = np.asarray(np.column_stack((data_test[:, 0], data_test[:, 1])))
	true_label = np.asarray(data_test[:, 2])

	clf = QuadraticDiscriminantAnalysis()
	clf.fit(X,y)

	pred_label = clf.predict(X_test)

	#print(pred_label)


	print(accuracy_score(true_label, pred_label, normalize = True))






	
def qda_analysis(data_train, data_test):
	X = np.asmatrix(np.column_stack((data_train[:, 0], data_train[:, 1])))
	X_test = np.asmatrix(np.column_stack((data_test[:, 0], data_test[:, 1])))
	true_label = np.asarray(data_test[:, 2])
	#print(true_label)
	
	#class_pos = data_train[data_train[:, 2] == 1.0]
	class_pos = data_train[data_train[:, 2] == 1.0][:, np.array([True, True, False])]
	#print(class_pos.shape)
	
	#class_neg = data_train[data_train[:, 2] == -1.0]
	class_neg = data_train[data_train[:, 2] == -1.0][:, np.array([True, True, False])]
	#print(class_neg)
	
	mean_class_pos = np.mean(class_pos, axis = 0)
	#print(mean_class_pos)
	
	mean_class_neg = np.mean(class_neg, axis = 0)
	#print(mean_class_neg)
	
	mean_class_pos_column = np.full((200,2), mean_class_pos)
	mean_class_neg_column = np.full((200,2), mean_class_neg)
	
	#mean_est = np.concatenate((mean_class_pos_column, mean_class_neg_column), axis = 0)
	#print(mean_est.shape)
	
	covariance_pos = np.cov(np.transpose(class_pos))
	#covariance_pos = np.cov(class_pos[:, 0], class_pos[:, 1])
	print(covariance_pos)
	
	covariance_neg = np.cov(np.transpose(class_neg))
	print(covariance_neg)
	
	func_of_x_pos = 0.5*(X_test - mean_class_pos_column)*np.linalg.inv(covariance_pos)*np.transpose(X_test - mean_class_pos_column)
	func_of_x_neg = 0.5*(X_test - mean_class_neg_column)*np.linalg.inv(covariance_neg)*np.transpose(X_test - mean_class_neg_column)
	
	#print(func_of_x_pos.shape)
	#print(func_of_x_neg.shape)
	
	
	func_of_x = (func_of_x_neg - func_of_x_pos)[:, 0] + 0.5*np.log(np.linalg.det(covariance_neg)) - 0.5*np.log(np.linalg.det(covariance_pos))
	
	#print(func_of_x)
	
	pred_label = np.where(func_of_x > 0, 1.00, func_of_x)
	pred_label = np.where(pred_label <= 0, -1.00, pred_label)
	#print(np.asarray(func_of_x).flatten())
	
	print(accuracy_score(true_label, pred_label, normalize = True))
	
	

def plot_after_classification(data_test, pred_label):

	#x = [data_test[:, 0], data_test[:, 2]]
	#y = [data_test[:, 1], data_test[:, 2]]

	true_label = data_test[:, 2]
	X_pos = []
	X_neg = []
	mis_classfied_pos = []
	mis_classfied_neg = []

	for i in range(200):
		if data_test.item((i, 2)) == 1.00:
			X_pos.append([data_test.item((i, 0)), data_test.item((i, 1))])
		else:
			X_neg.append([data_test.item((i, 0)), data_test.item((i, 1))])

		if data_test.item((i, 2)) == 1.00 and pred_label.item(i, 0) == -1.00:
			mis_classfied_pos.append([data_test.item((i, 0)), data_test.item((i, 1))])
		elif data_test.item((i, 2)) == -1.00 and pred_label.item(i, 0) == 1.00:
			mis_classfied_neg.append([data_test.item((i, 0)), data_test.item((i, 1))])

	#print(np.asmatrix(X_neg))

	X_pos = np.asarray(X_pos)
	X_neg = np.asarray(X_neg)
	mis_classfied_pos = np.asarray(mis_classfied_pos)
	mis_classfied_neg = np.asarray(mis_classfied_neg)

	#print(mis_classfied_neg)

	fig = plt.figure()
	plots = fig.add_subplot(111)

	plots.scatter(X_pos[:,0], X_pos[:,1], color='green', marker = '^', label = 'positive', alpha = 0.5)
	plots.scatter(X_neg[:,0], X_neg[:,1], color='blue', marker = 'x', label = 'negative', alpha = 0.5)
	if mis_classfied_pos.size!= 0:
		plots.scatter(mis_classfied_pos[:,0], mis_classfied_pos[:, 1], color='red', marker = 'o', label = 'mis-classified-pos', alpha = 0.5)
	if mis_classfied_neg.size!= 0:
		plots.scatter(mis_classfied_neg[:,0], mis_classfied_neg[:, 1], color='orange', marker = 's', label = 'mis-classified-neg', alpha = 0.5)

	plt.xlabel("x")
	plt.ylabel('y')
	plt.show()





	
def log_regression_library(data_train, data_test):

	X = np.column_stack((data_train[:, 0], data_train[:, 1]))
	y = data_train[:, 2]

	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

	X_test = np.column_stack((data_test[:, 0], data_test[:, 1]))

	pred_label = clf.predict(X_test)
	#print(pred_label)

	true_label = data_test[:, 2]
	
	print('Accuracy using library: ')
	#print(clf.score(X, y))
	print(accuracy_score(true_label, pred_label, normalize = True))


def logistic_regression(data_train, data_test):

	X = np.column_stack((data_train[:, 0], data_train[:, 1]))
	X = np.asmatrix(X)
	#print(X)

	train_label = np.asmatrix(data_train[:, 2])
	train_label = np.transpose(train_label)
	train_label = np.where(train_label == -1.0, 0, train_label)

	print(train_label.shape)

	#Lump a 1 into X
	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
	#print(X.shape)

	X_test = np.column_stack((data_test[:, 0], data_test[:, 1]))
	#Lump a 1 into X_test
	X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

	w = np.asmatrix([0.0, 0.0, 0.0])
	#w = np.asmatrix([1000, 1000, 1000])
	
	e = np.full((200, 1), 0.0001)
	
	#print(e)
	#print(e.shape)
	
	w_prev = np.asmatrix([0.0, 0.0, 0.0])

	w_diff = []

	w_delta = 1.00

	#for i in range(100):
	while w_delta > 0.005:
		exp_val = np.exp(w*np.transpose(X))

		exp_val_neg = np.exp((-1)*w*np.transpose(X))
		#print(exp_val)
		
		
		#print(exp_val)
		p = exp_val/ (1 + exp_val)

		#p = 1 / (1 + exp_val_neg)

		p = np.where(np.isinf(p), 1.0, p)

		p = np.transpose(p)
		#print(p)

		s = 1 - p
		#print(s.shape)
		#z = np.transpose(w*np.transpose(X))
		z = np.transpose(w*np.transpose(X)) + (train_label - p)/(s + e)
		#zz = (train_label - p)/(s + e)
		#print('z')
		#print(z.shape)

		S = np.diag(np.asarray(s).flatten())
		#print(S.shape)
		
		w = np.linalg.inv(np.transpose(X)*S*X)*np.transpose(X)*S*z
		w = np.transpose(w)
		w = w/np.linalg.norm(w)

		#print(w)

		w_delta = np.linalg.norm(w - w_prev)
		w_delta = np.asarray(w_delta).flatten()
		w_diff.append(w_delta)

		w_prev = w 

	#print(w.shape)

	#print(np.asarray(w_diff))
		
		
	print(np.asarray(w).flatten())
	w = np.asarray(w).flatten()
	
	
	#func_of_x = 1 / (1 + w*np.asmatrix(np.transpose(X_test)))
	func_of_x = 1 / (1 + np.exp((-1)*w*np.asmatrix(np.transpose(X_test))))
	func_of_x = np.transpose(func_of_x)
	#print(func_of_x)

	pred_label = np.where(func_of_x > 0.5, 1.0, func_of_x)
	pred_label = np.where(pred_label <= 0.5, -1.0, pred_label)

	#print(pred_label)

	true_label = data_test[:, 2]

	print(accuracy_score(true_label, pred_label, normalize = True))

	plot_after_classification(data_test, pred_label)





def main():

	np.random.seed(2022)
	data_train, data_test = generate_data()
	
	"""
	print('LDA Analysis .....\n')
	lda_analysis(data_train, data_test)
	print('LDA Analysis using library: ')
	lda_analysis_library(data_train, data_test)

	print('Logistic regression .....\n')
	logistic_regression(data_train, data_test)
	#print('Logistic regression using library: ')
	log_regression_library(data_train, data_test)
	
	#plt.show()"""
	qda_analysis(data_train, data_test)
	qda_using_library(data_train, data_test)


if __name__ == "__main__":
	main()

