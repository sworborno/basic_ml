import numpy as np
from sklearn import preprocessing, svm
import pandas as pd


def soft_margin_svm(X_train, y_train, X_test, y_test):
	opt_c = 0.0
	max_accuracy = 0.0
	for c in np.arange(0.001, 1.0, 0.001):
		clf = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
		score = clf.score(X_test, y_test)

		if score > max_accuracy:
			max_accuracy = round(score, 2)
			opt_c = c
	return max_accuracy, opt_c



def k_fold_svm(data, num_fold):

	folds = np.array_split(data, num_fold)
	for i in range(num_fold):

		test = folds[i]
		train = np.empty((0, 61), dtype=float)

		for j in range(num_fold):
			if not i == j:
				train = np.concatenate((train, folds[j]))

		X_train = train[:, 0 : -1]
		y_train = train[:, -1]

		X_test = test[:, 0 : -1]
		y_test = test[:, -1]

		max_accuracy, opt_c = soft_margin_svm(X_train, y_train, X_test, y_test)
		print('Maximum accuracy using fold {} = {}, with C = {}'.format(i, max_accuracy, opt_c))



def main():

	data = np.loadtxt('sonar5841.dat')
	X = data[:, 0 : -1]
	Y = data[:, -1]

	#print(X.mean(axis = 0))

	X = preprocessing.scale(X)
	#print(X.mean(axis = 0))

	data = np.column_stack((X, Y))
	#print(data[0])

	# Shuffle does not return anything, it shuffles the elements in the given array
	np.random.shuffle(data)

	#print(data[0])
	#print(data.shape)
	k_fold_svm(data, 5)


if __name__ == "__main__":
	main()
