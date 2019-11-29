import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as prepro
from mpl_toolkits.mplot3d import Axes3D
from cvxopt import matrix, solvers
from numpy import array, dot
from qpsolvers import solve_qp
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge

def generate_data():

	X = np.random.uniform(low = -10.0, high = 10.0, size = (3,200))
	#print('Dimension of X: '+ str(X.shape))
	W = np.array([[-0.8, 2.1, 1.5]])
	#print(W)
	mu = 0
	sigma = np.sqrt(10.0)
	epsilon = np.random.normal(mu, sigma, size = (1,200))
	#print(epsilon.shape)
	bias = 10
	Y = np.matmul( np.transpose(W), X) + bias + epsilon

	print(X.shape)
	print(Y.shape)

	return X, Y


def plot_data(X, Y):

	fig = plt.figure()
	axis = fig.add_subplot(111, projection = '3d')

	x1 = X[:,0]
	#print(x1.shape)
	x2 = X[:,1]
	#print(x2.shape)
	x3 = X[:,2]
	#print(x3.shape)
	c = Y[:,0]
	#print(c.shape)

	axis.scatter(x1, x2, x3, c = c, cmap = plt.hot())
	plt.show()


def standardize_data(X, Y):
	X_new = X

	for i in range(0, 3):
		X_new[i,:] = (X[i,:] - X[i,:].mean())/ X[i,:].std()
		#if np.mean(X_new[i,:]) < 0.000000001:

	Y = Y - np.mean(Y)
	#print(Y.mean())
	return X_new, Y



def plot_w_vs_lambda(w_list):
	lambda_array = []
	for l in np.arange(0.0, 0.5, 0.01):
		lambda_array.append(l)

	plt.scatter(np.asarray(lambda_array), np.asarray(w_list)[:, 0], c = 'r', marker = 's', label = 'w0', alpha = 0.5)
	plt.scatter(np.asarray(lambda_array), np.asarray(w_list)[:, 1], c = 'g', marker = 's', label = 'w1', alpha = 0.5)
	plt.scatter(np.asarray(lambda_array), np.asarray(w_list)[:, 2], c = 'b', marker = 's', label = 'w2', alpha = 0.5)

	plt.xlabel('lambda')
	plt.ylabel('Ws')
	plt.show()




def ridge_regression(X, Y):
        train_X = X[:100]
        train_Y = Y[:100]
        #print(train_Y)
        test_X = X[100:200]
        test_Y = Y[100:200]
        
        x_transpose_x = np.matmul(np.transpose(train_X), train_X)
        #print(x_transpose_x.shape)
        x_transpose_y = np.matmul(np.transpose(train_X), train_Y)
        #print(x_transpose_y.shape)
        y_predlist = []
        w_hat_list = []
        for lambdA in np.arange(0.0, 0.5, 0.01):
                lambda_I = lambdA*np.identity(3)
                #print(lambda_I)
                #w_hat = np.dot(np.linalg.inv(np.add(x_transpose_x, lambda_I)), x_transpose_y)
                w_hat = np.matmul(np.linalg.inv( x_transpose_x + lambda_I), x_transpose_y)
                #print(w_hat)
                w_hat_list.append(w_hat)
                #print(w_hat.shape)
                #print(test_X.shape)

                y_pred = np.matmul(test_X, w_hat) + np.mean(Y)
                print(y_pred)

                """
                error = y_pred - test_Y
                squared_error = np.square(error)
                sse = np.sum(squared_error)
                print(sse)
                """

                #y_predlist.append(y_pred)
        plot_w_vs_lambda(w_hat_list)
        #SSE = np.square(test_Y - np.asarray(y_predlist))
        #print(SSE)

def ridge_using_library(X, Y):
	train_X = X[:100]
	train_Y = Y[:100]
	test_X = X[100:200]
	test_y = Y[100:200]

	for lambdA in np.arange(0.01, 0.05, 0.01):
		ridge_reg = Ridge(alpha = lambdA, solver = "cholesky")
		ridge_reg.fit(train_X, train_Y)
		y_pred = ridge_reg.predict(test_X)
		print(y_pred)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

	P = .5 * (P + P.T)  # make sure P is symmetric
	args = [matrix(P), matrix(q)]
	if G is not None:
		args.extend([matrix(G), matrix(h)])
		if A is not None:
			args.extend([matrix(A), matrix(b)])
	sol = solvers.qp(*args)
	if 'optimal' not in sol['status']:
		return None
	return np.array(sol['x']).reshape((P.shape[1],))


def apply_tibshiranis_method(X, Y, t):

	
	H = np.asmatrix(2*np.matmul(np.transpose(X), X))
	f = np.asmatrix((-2)*np.matmul(np.transpose(X), Y))

	#First find the least square solution of w_ls = LS(X, Y)
	w_ls = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), Y))
        print(w_ls)

	matrix_A = []
	size_of_b = 0

	#print(np.linalg.norm(w_ls))
	while np.linalg.norm(w_ls) > t:
		#Determine the sign of w_ls
		lambda_i_0 = np.transpose(np.sign(w_ls)).tolist()

		#print(lambda_i_0.shape)
		matrix_A.append(lambda_i_0)

		#print(matrix_A)
		#print(np.asmatrix(np.asarray(matrix_A)).shape)
		size_of_b += 1

		b = np.ones(shape = (size_of_b, 1))
		A = np.asarray(matrix_A)
		print(A)
		#w_ls = cvxopt_solve_qp(H, f, np.asarray(matrix_A), b)
		w_ls += 1

	print(w_ls)

	
	
	

def lasso_regression(X, Y):
	train_X = X[:100]
	train_Y = Y[:100]
	test_X = X[100:200]
	test_Y = Y[100:200]


	train_X, train_Y = standardize_data(train_X, train_Y)
	test_X, test_Y = standardize_data(test_X, test_Y)


	#Least square solution to w
	#w_ls = np.matmul(np.linalg.inv(np.matmul(np.transpose(train_X), train_X)), np.matmul(np.transpose(train_X), train_Y))
	#print(w_ls)

	apply_tibshiranis_method(train_X, train_Y, (1.0/0.1))


	"""
	H = np.asmatrix(2*np.matmul(np.transpose(train_X), train_X))
	f = np.asmatrix((-2)*np.matmul(np.transpose(train_X), train_Y))

	#print(H.shape)
	#print(f.shape)

	A = np.array([[-1, -1, -1], 
		[-1, -1,  1],
		[-1,  1, -1],
		[-1,  1,  1],
		[ 1, -1, -1],
		[ 1, -1,  1],
		[ 1,  1, -1],
		[ 1,  1,  1]])

	A = matrix(A, (8, 3), 'd')
	for lambdA in np.arange(0.01, 0.05, 0.01):
		b = np.full(shape = (8,1), fill_value = 1/lambdA)
		sol = cvxopt_solve_qp(H, f, A, b)
		sol = np.transpose(np.asmatrix(sol))
		y_pred = np.matmul(test_X, sol) + np.mean(Y)
		print(y_pred)
	"""


def lasso_using_library(X, Y):
	train_X = X[:100]
	train_Y = Y[:100]
	test_X = X[100:200]
	test_y = Y[100:200]

	train_X, train_Y = standardize_data(train_X, train_Y)
	test_X, test_Y = standardize_data(test_X, test_Y)

	for lambdA in np.arange(0.01, 0.05, 0.01):
		lasso_reg = Lasso(alpha = lambdA)
		lasso_reg.fit(train_X, train_Y)
		y_pred = lasso_reg.predict(test_X)
		print(y_pred)





def main():

	np.random.seed(2019)
	X, Y = generate_data()

	#print(Y)
	#plot_data(X, Y)
	#X, Y = standardize_data(X,Y)
	
	#ridge_regression(X, Y)
	#print('*******************')
	#ridge_using_library(X, Y)

	#lasso_regression(X,Y)
	#print('*******************')
	#lasso_using_library(X, Y)


if __name__ == "__main__":
	main()
